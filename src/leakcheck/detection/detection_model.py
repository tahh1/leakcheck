from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain.chat_models import init_chat_model
from src.leakcheck.llm.prompts import prompts
from src.leakcheck.llm.PreprocEngine import PreprocFSEngine
from src.leakcheck.llm.OverlapEngine import OverlapFSEngine
from leakcheck.configs import configs
from pydantic import BaseModel, Field
from typing_extensions import List
import json


load_dotenv()

class ExplanationObject(BaseModel):
    line_number: int = Field(description="Line number where a leaky transformation was performed.")
    explanation: str = Field(description="Explanation of the leaky transformation")

class DetectionObject(BaseModel):
    leakage_detected : bool = Field(description="The leakage label")
    leakage_lines : List[ExplanationObject] = Field(
        default_factory=list,
        description="A list of objects showing the lines where leaky transformations are performed along with their explanation.")
    


def detect(analysis_record,leak,num_shots):
    
    
    model = init_chat_model(model=configs.model, api_key= os.getenv('api_key'))
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
        
        print(f"Classifying pair {pair} ...")
        snippet = pair["snippet"]
        model_info = pair["model info"]
        few_shot_examples = fs_engine.choose_examples(snippet,"pattern-match",num_shots)
    
        messages_to_pass = []
        for i,shot in few_shot_examples.iterrows():
            messages_to_pass.extend([
                HumanMessage(content=f"model info:\n{shot["Model info"]}\nsnippet:\n{shot["Leak"]}"),
                AIMessage(content=json.dumps(shot["New explanation"]))])
        
        messages_to_pass.append(HumanMessage(content=f"model info:\n{model_info}\nsnippet:\n{snippet}")) 
        
        input = prompt_template.invoke({"few-shots":messages_to_pass})
        
    
        detection_results = model_with_structured_output.invoke(input)
        print(f"{leak}_Leakage detected: ",detection_results.leakage_detected)
        print(f"{leak}_Explanation: ", [
            {"line_number":expl.line_number, "explanation":expl.explanation} for expl in
            detection_results.leakage_lines])
        
        
        pair[f"{leak}_leakage_detected"] = detection_results.leakage_detected
        pair[f"{leak}_leakage_lines"] = [
            {"line_number":expl.line_number, "explanation":expl.explanation} for expl in
            detection_results.leakage_lines]
        

    return analysis_record









