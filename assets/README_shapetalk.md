# Information about the two .csv files containing ShapeTalk

## $${\color{blue}(A) \space shapetalk \textunderscore \color{red} raw \color{blue} \textunderscore public \textunderscore version \textunderscore 0.csv}$$

The first file (language/shapetalk/) **shapetalk_raw_public_version_0.csv** contains *20 columns and 130,342 rows*, which in aggregate contain 536,596 discriminative utterances concerning pairs of 3D objects.

With Python you can quickly load it as:

```{python}
import pandas
df = pandas.read_csv("language/shapetalk/shapetalk_raw_public_version_0.csv")
print(len(df))
df.head()
```

The 20 columns' names, along with explanations about their content, are given below:

| Column | Description |
| --- | --- |
| workerid | Unique anonymized identifier corresponding to each individual annotator|
| utterance_0-4| At *maximum*, we asked for 5 descriptions for a pair, denoted here as utterance_[0-4] |
||note: When less than 5 descriptions were collected, their absence is marked with None/NaN|.
|assignmentid| Unique identifier corresponding to the responses of an annotator for a specific communication context (source, target pair).|
|worktimeinseconds|Seconds it took to submit all [0-5] utterances of the underlying communication context by a given annotator.|

Columns regarding identifying the 3D model used as a *source* (or equivalently as a `distractor') in a given communication context. Recall: the descriptions collected are meant to distinguish each *target* object of the pair from this distracting one.
| Column | Description |
| --- | --- |
|source_model_name| the modelname, typically a long string|
|source_object_class| object-class *according to ShapeTalk's* class naming conventions (see: changeit3d/in_out/datasets/shape_talk.py)
|source_dataset| either ShapeNet, PartNet, or ModelNet|
|source_original_object_class| the object-class as in the **original** dataset listed above (i.e., not following ShapeTalk's convention; for convention see [F.A.Q.](https://github.com/optas/changeit3d/#frequently-asked-questions-faq)|
|source_uid | a unique-id for each model made like this: *source_object_class/source_dataset/source_model_name*|
||note: this is also how we store each model on the downloadable folder containing ShapeTalk's data (e.g., /point_clouds/<source_uid>.npz)|
|source_unary_split|indicates if the source object of the underlying pair/row was in train or test or val set. |
||note: This should be used when training auxiliary networks that operate on **single input objects**, such as a PC-AE, SGF-AE, etc.|

Next, there are 6 equivalent columns like the above about the **target** object of each communication context (target_model_name,              target_object_class, target_dataset, target_original_object_class, target_uid, target_unary_split)

Finally,
| Column | Description |
| --- | --- |
|is_patched| Boolean, indicates if one or more of the underlying utterances were manually patched to improve some aspect of them (e.g., spelling) following communication between the annotators and the authors
|hard_context| Boolean, is the underlying paired source/target visually similar and hence a hard context, or not?|

## $${\color{blue}(B) \space shapetalk \textunderscore \color{red} preprocessed \color{blue} \textunderscore public \textunderscore version \textunderscore 0.csv}$$

This second file (language/shapetalk/) **shapetalk_preprocessed_public_version_0.csv**  is a preprocessed version of the above "raw" file.
Its main differences compared to the raw file are:

1. contains more (train/test) splits used in our work for training neural listeners or language-assisted 3D editors (changeIt3D task)
2. the inclusion of further *spell-checked* ShapeTalk utterances (which is what we used for our paper)
3. the inclusion of the derived *tokens* (again as used by our paper's methods)
4. the separation of the "raw" utterances ([0-4]) corresponding to a given 3D object pair, into distinct rows. This operation boosts the size of the "raw" file above (130,342 rows) to 536,596 ("preprocessed") rows.

Specifically,

| (extra or new) Columns | Description |
| --- | --- |
|utterance| Each row contains only one of the "raw" utterances_0-4 for each pair|
|saliency| integer that indicates which of the 0-4 utterances of the "raw" file each row contains|
||note: the 0-th utterance (saliency) was given by the annotator first. then the 1st-utterance was given etc. Typically, the earlier the utterance was given by an annotator; the more pronounced (salient) its description is (see listening experiments of our paper)|
|utterance_spelled| the utterance after basic canonicalization (e.g., lower-casing) and spell-checkers have been applied to it (**this what we use for our paper/experiments**)|
|tokens| the tokens (list of integers) corresponding taken from the utterance_spelled|
|tokens_encoded| the tokens converted to integers using the (can-be-dowloaded-with-shapetalk-data) vocabulary.pkl|
|tokens_len| integer, the length of the tokens|
|listening_split| split used to train/test/validate our neural listeners|
||values={**_train_,_test_,_val_,_ignore_**} (`ignore' utterances with more than 16 tokens (99th percentile))|
|changeit_split| split used to train/test/validate our ChangeIt3D network(s)|
||note: The input to a ChangeIt3D system is the *distractor* of each pair, with the underlying human language created for its target so as to expect/make the distractor object look more like the target.|
