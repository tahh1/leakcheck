from __future__ import annotations

from leakcheck.llm.FewShotEngine import BaseLeakageFewShotEngine


# Define sets of known methods based on interpretation
# These sets would ideally be refined with user guidance or tags in the original data
# Methods expected to be used directly as function calls
callables_overlap = {
    'sample',
        # Data augmentation / transformations (torchvision, keras, albumentations)
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "RandomResizedCrop",
    "RandomAffine",
    "RandomPerspective",
    "RandomGrayscale",
    "RandomInvert",
    "RandomPosterize",
    "RandomSolarize",
    "RandomAdjustSharpness",
    "RandomAutocontrast",
    "RandomEqualize",
    "Compose",
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotate90",
    "ShiftScaleRotate",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "CLAHE",
    "RandomCrop",
    "CenterCrop",
    "Resize",
    "Normalize",
}



instantiated_classes_overlap = {
    # Resampling
    "RandomOverSampler", "SMOTE", "ADASYN", "RandomUnderSampler", "KMeansSMOTE", "NearMiss",
    "TomekLinks", "EditedNearestNeighbours", "Pipeline", "BorderlineSMOTE", "SVMSMOTE",
    # Augmentation
    "ImageDataGenerator",
}

class OverlapFSEngine(BaseLeakageFewShotEngine):
    def __init__(self,df_path) -> None:
        super().__init__(
            df_path=df_path,
            callables=callables_overlap,
            classes=instantiated_classes_overlap,
            usage_templates=[
                r'{var}\.fit_resample\s*\(',
                r'{var}\.fit\s*\(',
                r'{var}\.sample\s*\(',
                r'{var}\.choice\s*\(',
            ],
            extra_detectors=[],
            equivs={},
            empty_fallback=self._overlap_empty_fallback,
        )
        
    def _overlap_empty_fallback(self,n):
        # Keep original behavior: select exactly 3 items from chosen sub-clusters,
        # regardless of requested n.
        chosen_subcluster = ["Oversampling/resampling", "Split errors", "Train as Test"]
        filtered = self.df[self.df['Sub Cluster'].isin(chosen_subcluster)]
        size = min(3, len(filtered))
        if size == 0:
            return super().diversified_sampling(self.df, n)
        return filtered.sample(n=size)

