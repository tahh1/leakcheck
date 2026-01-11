from __future__ import annotations
import random
import pandas as pd
import re



class BaseLeakageFewShotEngine:
    def __init__(
        self,
        df_path,
        *,
        callables: set,
        classes: set,
        usage_templates: list,
        extra_detectors: list | None = None,
        equivs: dict | None = None,
        empty_fallback=None,
    ) -> None:
        self.df = pd.read_csv(df_path)
        self.callables = set(callables)
        self.classes = set(classes)
        self.usage_templates = list(usage_templates)
        self.extra_detectors = list(extra_detectors or [])
        self.equivs = dict(equivs or {})
        self.empty_fallback = empty_fallback or self.diversified_sampling

    def match(self, snippet: str):
        return self.match_methods_with_regex(
            snippet,
            callables=self.callables,
            classes=self.classes,
            usage_templates=self.usage_templates,
            extra_detectors=self.extra_detectors,
            equivs=self.equivs,
        )

    def sample(self, matched_methods: list, n: int) -> pd.DataFrame:
        return self.hierarchical_sample_round_robin_exact(
            matched_methods,
            n,
            diversified_fallback=self.diversified_sampling,
            empty_fallback=self.empty_fallback,
        )

    def choose_examples(self,snippet: str, strategy: str, n: int) -> pd.DataFrame:
        if strategy == "random":
            ids = random.sample(self.df["Id"].tolist(), n)
            return self.df.set_index("Id").loc[ids].reset_index()
        elif strategy == "pattern-match":
            matched = self.match(snippet) 
            print(f"Pattern matched: {matched}")
            return self.sample(matched, n)
        else:
            raise ValueError("Invalid strategy")
        
    def build_instantiation_and_usage_patterns(self,class_name, usage_templates):
        # Allow optional module prefix (e.g. decomposition.PCA or sklearn.decomposition.PCA)
        qualified_class_pattern = rf'(?:\w+\.)*{re.escape(class_name)}'

        # Match var assignment like: pca = decomposition.PCA(...)
        var_pattern = rf'(\w+)\s*=\s*{qualified_class_pattern}\s*\('

        # Match inline usage like: decomposition.PCA(...).fit_transform(...)
        inline_pattern = rf'{qualified_class_pattern}\s*\([^\)]*\)\s*\.\s*\w+\s*\('

        return var_pattern, usage_templates, inline_pattern
    
    
    def build_callable_pattern(self,method):
        return rf'\b{re.escape(method)}\s*\('


    def match_methods_with_regex(self,
        snippet,
        *,
        callables=None,
        classes=None,
        usage_templates=None,
        extra_detectors=None,
        equivs=None,
    ):
        callables = callables or set()
        classes = classes or set()
        usage_templates = usage_templates or []
        extra_detectors = extra_detectors or []

        matched = set()

        # Match callable method patterns
        for method in callables:
            if re.search(self.build_callable_pattern(method), snippet):
                matched.add(method)

        # Extra detectors (e.g., SelectFromModel variants)
        for det in extra_detectors:
            try:
                detected = det(snippet)
                if detected:
                    matched.update(detected)
            except Exception:
                pass

        # Match class instantiation + usage
        for class_name in classes:
            var_assign_pattern, ut, inline_pattern = self.build_instantiation_and_usage_patterns(class_name, usage_templates)

            match = re.search(var_assign_pattern, snippet)
            if match:
                var_name = match.group(1)
                for template in ut:
                    usage_regex = template.replace('{var}', re.escape(var_name))
                    if re.search(usage_regex, snippet):
                        matched.add(class_name)
                        break

            if re.search(inline_pattern, snippet):
                matched.add(class_name)

        # Deduplicate with equiv mapping
        result = []
        seen = set()
        for x in matched:
            value = equivs.get(x, x) if equivs else x
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result



    def diversified_sampling(self,n):
        """
        Fully diversified sampling by flattening and shuffling all (cluster, sub-cluster, method) combos.
        Performs round-robin sampling across the full list to ensure cluster diversity.
        """
        result = pd.DataFrame()
        used_indices = set()
        buckets = []

        # Create a flat list of (cluster, sub-cluster, method) groupings
        combos = self.df[["Cluster", "Sub Cluster", "Method"]].dropna().drop_duplicates().values.tolist()
        random.seed(42)
        random.shuffle(combos)

        # Create buckets for each unique (cluster, sub-cluster, method)
        for cluster, subcluster, method in combos:
            group_rows = self.df[
                (self.df["Cluster"] == cluster) &
                (self.df["Sub Cluster"] == subcluster) &
                (self.df["Method"] == method) &
                (~self.df.index.isin(used_indices))
            ]
            if not group_rows.empty:
                buckets.append(group_rows)

        # Round-robin through buckets
        while len(result) < n and buckets:
            next_buckets = []
            for bucket in buckets:
                row = bucket.head(1)
                if not row.empty:
                    result = pd.concat([result, row])
                    used_indices.update(row.index)
                    remaining = bucket.iloc[1:]
                    if not remaining.empty:
                        next_buckets.append(remaining)
                if len(result) >= n:
                    break
            buckets = next_buckets

        return result.head(n).reset_index(drop=True)


    def hierarchical_sample_round_robin_exact(self,matched_methods, n, diversified_fallback, empty_fallback=None):
        if not matched_methods:
            fallback = empty_fallback or diversified_fallback
            return fallback(n).reset_index(drop=True)

        exact_matches = pd.DataFrame()
        fallback_matches = pd.DataFrame()
        used_indices = set()

        method_index = 0
        rounds = 0
        max_rounds = 10  # safety
        while (len(exact_matches) + len(fallback_matches)) < n and rounds < max_rounds:
            method = matched_methods[method_index % len(matched_methods)]

            # Step 1: Exact Method match
            available = self.df[(self.df["Method"] == method) & (~self.df.index.isin(used_indices))]
            to_add = available.head(1)

            if not to_add.empty:
                exact_matches = pd.concat([exact_matches, to_add])
                used_indices.update(to_add.index)
            else:
                # Step 2: Sub-cluster fallback
                subclusters = self.df[self.df["Method"] == method]["Sub Cluster"].dropna().unique()
                sub_match = self.df[(self.df["Sub Cluster"].isin(subclusters)) & (~self.df.index.isin(used_indices))]
                to_add = sub_match.head(1)

                # Step 3: Cluster fallback
                if to_add.empty:
                    clusters = self.df[self.df["Method"] == method]["Cluster"].dropna().unique()
                    cluster_match = self.df[(self.df["Cluster"].isin(clusters)) & (~self.df.index.isin(used_indices))]
                    to_add = cluster_match.head(1)

                if not to_add.empty:
                    fallback_matches = pd.concat([fallback_matches, to_add])
                    used_indices.update(to_add.index)

            method_index += 1
            if method_index >= len(matched_methods):
                method_index = 0
                rounds += 1

        # Step 4: Diversified sampling
        total_selected = len(exact_matches) + len(fallback_matches)
        if total_selected < n:
            remaining_needed = n - total_selected
            #filler = diversified_fallback(self.df[~self.df.index.isin(used_indices)], remaining_needed)
            filler = diversified_fallback(remaining_needed)
            fallback_matches = pd.concat([fallback_matches, filler])

        # ✅ Exact matches always come first
        final = pd.concat([exact_matches, fallback_matches]).head(n).reset_index(drop=True)
        return final




    

