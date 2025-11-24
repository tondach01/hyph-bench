# Hyph-bench: Benchmark Dataset of Hyphenated Words for Generating Hyphenation Patterns

version 1.0

Raw data and scripts for creating and evaluating hyphenated word lists in several languages.

## Repository structure

### data/
Directory with initial versions of non-Wiktionary datasets

### Makefile
Definition of helpful batch commands (`*` = `wikt` for Wiktionary datasets / `other` for other datasets):
`process_wikt`: unzip Wiktionary dump files and process them into initial word lists, which are stored in `data/`
`prepare_*`: perform initial preprocessing
`disambiguate_*`: eliminate ambiguous hyphenations
`translate_*`: create translate files necessary for **patgen** program
`stats_all_datasets`: compile statistics of all datasets
`cross_validate_all`: perform 10-fold cross-validation over all datassets with baseline profiles

### profiles/
Baseline parameter profiles.

### scripts/
Python scripts and packages used for data preprocessing, evaluation, and reporting.

### wikt_dump.zip
Compressed directory with JSON dump files of Wiktionary datasets.

## Licences
Please acknowledge the allowed usage of the data:

| Dataset               | Licence          | Note |
|-----------------------|------------------|------|
| cs/cshyphen\_cstenten |  CC BY-NC-SA 3.0 | |
| cs/cshyphen\_ujc      | MIT              | |
| cs/wiktionary         | CC BY-SA 4.0     | |
| cssk/cshyphen         | MIT              | |
| de/wiktionary         | CC BY-SA 4.0     | |
| de/wortliste          | MIT              | |
| el/wiktionary         | CC BY-SA 4.0     | |
| es/wiktionary         | CC BY-SA 4.0     | |
| is/hyphenation-is     | CC BY 4.0        | |
| it/wiktionary         | CC BY-SA 4.0     | |
| ms/wiktionary         | CC BY-SA 4.0     | |
| nl/wiktionary         | CC BY-SA 4.0     | |
| pl/wiktionary         | CC BY-SA 4.0     | |
| pt/wiktionary         | CC BY-SA 4.0     | |
| ru/wiktionary         | CC BY-SA 4.0     | |
| th/orchid             | CC BY-SA 4.0     | licensed in 2025 by Sojka from public domain |
| tr/wiktionary         | CC BY-SA 4.0     | |
| uk/wiktionary         | CC BY-SA 4.0     | from Sojka, O.: Transfer Learning of Slavic Syllabification for Hyphenation Patterns (Feb 2025), Bachelor Thesis, Masaryk University, Brno, Faculty of Informatics (advisor: Pavel Šmerk) |
