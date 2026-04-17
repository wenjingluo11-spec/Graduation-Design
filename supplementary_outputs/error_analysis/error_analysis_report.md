# Error Analysis Report

## sentiment_error_summary
- Source: `F:\python testing\Graduation Design\情感二分类\outputs_smoke\sentiment_error_summary.csv`

```text
best_model,error_type,count
NaiveBayes,correct,1283
NaiveBayes,false_negative,114
NaiveBayes,false_positive,103
```

## reuters_top_confusions
- Source: `F:\python testing\Graduation Design\新闻多分类\outputs_smoke\reuters_top_confusions.csv`

```text
model,true_label,pred_label,count
BiGRU,18,4,20
BiGRU,3,4,18
BiGRU,15,4,15
BiGRU,1,4,12
BiGRU,10,1,12
BiGRU,12,4,9
BiGRU,18,1,8
BiGRU,18,3,8
BiGRU,4,3,7
BiGRU,7,3,6
BiGRU,15,3,6
BiGRU,19,4,6
```

## translation_error_summary
- Source: `F:\python testing\Graduation Design\机器翻译\outputs_smoke\translation_error_summary.csv`

```text
model_decode,error_tag,count
seq2seq_beam,semantic_or_grammar_error,10
seq2seq_beam,low_keyword_overlap,2
seq2seq_greedy,semantic_or_grammar_error,9
seq2seq_greedy,low_keyword_overlap,3
transformer_beam,semantic_or_grammar_error,7
transformer_beam,under_translation,4
transformer_beam,low_keyword_overlap,1
transformer_greedy,semantic_or_grammar_error,9
transformer_greedy,low_keyword_overlap,3
```
