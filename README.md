# JTTW_Overseas_Reception_Analysis
Code for sentiment analysis and LDA topic modeling of English reviews on *Journey to the West* 

## 1. Dependencies Setup
### Step 1: Install Python
Recommended version: Python 3.8–3.12  
Download link: https://www.python.org/downloads/ (check "Add Python to PATH" during installation)

### Step 2: Install Required Packages
Open Command Prompt (Windows) / Terminal (Mac/Linux), navigate to the code folder, and run:
```bash
pip install pandas gensim nltk pyLDAvis
```

## 2. File Description
| Filename                          | Function                                      |
|-----------------------------------|-----------------------------------------------|
| S1_Code_Sentiment_Analysis.py     | Sentiment analysis code (merged lexicon method) |
| S2_Code_LDA.py                    | LDA topic modeling & visualization code       |
| AFINN-111.txt                     | General sentiment lexicon (AFINN)             |
| NRC-Emotion-Lexicon-Wordlevel-v0.92.txt | General sentiment lexicon (NRC)          |
| custom_lexicon.txt                | Custom domain lexicon for *Journey to the West* reviews |
| sample_data.xlsx                  | Sample review text (for code validation)      |

## 3. How to Run
1. **Sentiment Analysis**: Run `S1_Code_Sentiment_Analysis.py` directly; it will output sentiment scores and labels (positive/negative/neutral).
2. **LDA Topic Modeling**: Run `S2_Code_LDA.py` directly; it will output topic results and generate an interactive LDA visualization HTML file.
