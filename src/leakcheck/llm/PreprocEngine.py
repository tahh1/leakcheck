from __future__ import annotations

from leakcheck.llm.FewShotEngine import BaseLeakageFewShotEngine
import re
callables_preproc = {
    'fillna', 'replace', 'cut', 'qcut', 'clip', 'get_dummies',
    'quantile_transform', 'scale', 'mean', 'std',
    'min', 'max', 'groupby', 'agg', 'to_dict', 'transform'
}

equivs = {
    'mean': 'Manual standardization (using mean() and std())',
    'std': 'Manual standardization (using mean() and std())',
    'min': 'Manual normalization (using min() and max())',
    'max': 'Manual normalization (using min() and max())',
    'scale': 'Preprocessing.scale()',
    'replace': 'df.replace(np.nan, df.mean())',
    'agg': 'df.groupby().agg().to_dict() →map()',
    'to_dict': 'df.groupby().agg().to_dict() →map()',
    'groupby': 'df.groupby().transform()',
    'SelectFromModel-prefit-true': 'SelectFromModel(prefit=True)',
    'SelectFromModel-prefit-false': 'SelectFromModel(prefit=False)',
    'LogisticRegression':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'Lasso':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'Ridge':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'ElasticNet':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'LinearSVC':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'SVC':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'DecisionTreeClassifier':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'RandomForestClassifier':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"',
    'XGBoostClassifier':'"(LogisticRegression, Lasso, \r\nRidge|ElasticNet|LinearSVC|\r\n SVC(kernel=""linear"")).coef_, \r\n(DecisionTreeClassifier|Rand\r\n omForestClassifier|XGBoostC\r\n lassifier).feature_importanc\r\n es_"'
}

instantiated_classes_preproc = {
    'SimpleImputer', 'MeanMedianImputer', 'EndTailImputer', 'CategoricalImputer',
    'Regression Imputation', 'KNNImputer', 'Autoencoder-Based Imputation',
    'iterativeImputer', 'RandomSampleImputer',
    'KBinsDiscretizer', 'EqualFrequencyDiscretiser', 'EqualWidthDiscretiser',
    'DecisionTreeDiscretiser', 'LDA', 'PCA', 'CountVectorizer', 'HashingVectorizer',
    'Tfidf Transformer', 'TfidfVectorizer', 'SelectKBest', 'VarianceThreshold', 'RFE',
    'Sequential Feature Selection', 'SelectFromModel', 'ELI5', 'LogisticRegression',
    'Lasso', 'Ridge', 'ElasticNet', 'LinearSVC', 'SVC', 'DecisionTreeClassifier',
    'RandomForestClassifier', 'XGBoostClassifier', 'OneHotEncoder', 'LabelEncoder',
    'CountFrequencyEncoder', 'RareLabelEncoder', 'DecisionTreeEncoder',
    'CatBoostEncoder', 'CountEncoder', 'GLMMEncoder', 'JamesSteinEncoder',
    'LeaveOneOutEncoder', 'MEstimateEncoder', 'StringLookup', 'IntegerLookup',
    'CategoryEncoding', 'WOEEncoder', 'MeanEncoder', 'QuantileEncoder',
    'TargetEncoder', 'StandardScaler', 'RobustScaler', 'MinMaxScaler',
    'maxabs_scale', 'TfidfTransformer'
}



class PreprocFSEngine(BaseLeakageFewShotEngine):
    def __init__(self,df_path) -> None:
        super().__init__(
            df_path=df_path,
            callables=callables_preproc,
            classes=instantiated_classes_preproc,
            usage_templates=[
                r'{var}\.fit_transform\s*\(',
                r'{var}\.transform\s*\(',
                r'{var}\.coef_\b',
                r'{var}\.feature_importances_\b',
            ],
            extra_detectors=[self.detect_selectfrommodel_variants],
            equivs=equivs,
            empty_fallback=self.diversified_sampling,
        )

    def match(self, snippet: str):
        # Avoid double-matching SelectFromModel while keeping its variant detector
        classes = set(self.classes)
        if 'SelectFromModel' in classes:
            classes.remove('SelectFromModel')
        return self.match_methods_with_regex(
            snippet,
            callables=self.callables,
            classes=classes,
            usage_templates=self.usage_templates,
            extra_detectors=self.extra_detectors,
            equivs=self.equivs,
        )
        
    def detect_selectfrommodel_variants(self,snippet):
        matches = re.finditer(r'SelectFromModel\s*\(([^)]*)\)', snippet)
        detected = set()
        for match in matches:
            args = match.group(1).replace(" ", "")
            if 'prefit=True' in args:
                detected.add('SelectFromModel-prefit-true')
            else:
                detected.add('SelectFromModel-prefit-false')  # includes missing prefit (defaults to False)
        return detected

