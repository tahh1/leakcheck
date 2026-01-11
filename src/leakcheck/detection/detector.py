import logging
import os
import json
from pandas import pd
from dotenv import load_dotenv
from typing_extensions import List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from leakcheck.configs import configs
from leakcheck.llm.prompts import prompts
from leakcheck.llm.PreprocEngine import PreprocFSEngine
from leakcheck.llm.OverlapEngine import OverlapFSEngine

logger = logging.getLogger(__name__)



load_dotenv()

class ExplanationObject(BaseModel):
    line_number: int = Field(description="Line number where a leaky transformation was performed.")
    explanation: str = Field(description="Explanation of the leaky transformation")

class DetectionObject(BaseModel):
    leakage_detected : bool = Field(description="The leakage label")
    leakage_lines : List[ExplanationObject] = Field(
        default_factory=list,
        description="A list of objects showing the lines where leaky transformations are performed along with their explanation.")
    
    
    
    
    
def build_fewshot_conversation(
    prompt_template : ChatPromptTemplate, 
    snippet: str, 
    model_info: str, 
    few_shot_examples: pd.DataFrame):
    
    
    messages_to_pass = []
    for _,shot in few_shot_examples.iterrows():
        messages_to_pass.extend([
                HumanMessage(content=f"model info:\n{shot["Model info"]}\nsnippet:\n{shot["Leak"]}"),
                AIMessage(content=json.dumps(shot["New explanation"]))])
        
    messages_to_pass.append(HumanMessage(content=f"model info:\n{model_info}\nsnippet:\n{snippet}")) 
        
    input = prompt_template.invoke({"few-shots":messages_to_pass})
    return input


def detect_leakage(
    analysis_record: List[dict],
    leak: str,
    num_shots: int =3):
    
    logger.info("Running %s leakage detection for %d snippets", leak, len(analysis_record))
    model = init_chat_model(model=configs.model, 
                            api_key= os.getenv('api_key'))
    
    model_with_structured_output = model.with_structured_output(DetectionObject)
    
    if leak == "preproc":
        fs_engine = PreprocFSEngine(configs.preproc_fs_path)
    else:
        fs_engine = OverlapFSEngine(configs.overlap_fs_path)
        
    prompt_template = ChatPromptTemplate([
            ("system", prompts[f"{leak}_detection"]),
            MessagesPlaceholder("few-shots")
        ],
            template_format="jinja2",
        )
        
    for pair in analysis_record:
        
        snippet = pair["snippet"]
        model_info = pair["model info"]
        few_shot_examples = fs_engine.choose_examples(snippet,
                                                      "pattern-match",
                                                      num_shots)
    
        input = build_fewshot_conversation(prompt_template, 
                                           snippet, 
                                           model_info, 
                                           few_shot_examples)
        
    
        detection_results = model_with_structured_output.invoke(input)

        pair[f"{leak}_leakage_detected"] = detection_results.leakage_detected
        
        pair[f"{leak}_leakage_lines"] = [
            {"line_number":expl.line_number, "explanation":expl.explanation} 
            for expl in detection_results.leakage_lines]
        

    return analysis_record










