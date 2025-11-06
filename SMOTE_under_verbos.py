from sklearn.neighbors import KNeighborsClassifier
import spacy 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from collections import Counter
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Carregar modelo spacy para português
nlp = spacy.load("pt_core_news_sm")

data = pd.read_csv ("factnews_dataset.csv", delimiter = ',')
df = pd.DataFrame (data)

print (df)

# Drop de colunas não necessárias 
X = df.drop(columns = ["file", "classe", "domain", "id_article"])

# y é a coluna de classes (target)
y = df.iloc[:, -1]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

## Criação dos dataframes de Treino ##
train_df = X_train.copy()
train_df["classe"] = y_train.values

# Contagem de amostras por classe nos dados de treino
print(f"\nDados de treino:", train_df["classe"].value_counts())

## Criação dos dataframes de Teste ##
test_df = X_test.copy()
test_df["classe"] = y_test.values

# Contagem de amostras por classe nos dados de teste
print(f"\nDados de teste:", test_df["classe"].value_counts())

print (train_df.shape)
print (test_df.shape)

# -- FUNÇÃO DE LIMPEZA E TOKENIZAÇÃO -- #
def limpar_tokenizar(texto):
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) != 1]
    return tokens

# Adiciona coluna de tokens aos Dataframes de treino e teste
train_df["tokens"] = train_df["sentences"].apply(limpar_tokenizar)
test_df["tokens"] = test_df["sentences"].apply(limpar_tokenizar)

# Criar vectorizer para bigramas
vectorizer = CountVectorizer(ngram_range = (2, 2), token_pattern = r'\b\w+\b', min_df = 1)

# Adicionar docs spaCy (para extrair adjetivos)
train_df["doc"] = list(nlp.pipe(train_df["sentences"], batch_size = 50))
test_df["doc"] = list(nlp.pipe(test_df["sentences"], batch_size = 50))

# -- CRIAÇÃO DO FICHEIRO (CHAVES: PALAVRAS, VALORES: PoS e Polaridade da palavra) -- #
def ficheiro_sentilex ():
    sentimentos = {}
    with open ("SentiLex.csv", encoding = "utf-8") as f:
        for line in f:
            line = line.strip ()
            parts = line.split(",")
            palavra = parts [0]
            POL = parts [5].split('=')[1]
            sentimentos[palavra] = POL

    return sentimentos

sentimentos = ficheiro_sentilex ()

# -- CONTAGEM DE PALAVRAS NEGATIVAS, NEUTRAS E POSITIVAS NUMA FRASE -- #
def contagem_por_polaridade (sentimentos, linha):
    positividade = 0
    negatividade = 0
    neutralidade = 0
    palavras_negativas = Counter()
    palavras_positivas = Counter()
    palavras_neutras = Counter()
    for word in linha.split ():
        word = word.lower()
        if word in sentimentos:
            POL = sentimentos[word]
            if POL == "-1":
                negatividade += 1
                palavras_negativas[word] += 1
            elif POL == "0":
                neutralidade += 1
                palavras_neutras[word] += 1
            else:
                positividade += 1
                palavras_positivas[word] += 1
    
    return negatividade,neutralidade, positividade, palavras_negativas, palavras_positivas, palavras_neutras

def palavras_polaridade_top (classe):
    todas_palavras_negativas = Counter()
    todas_palavras_positivas = Counter()
    todas_palavras_neutras = Counter()
    freq_neg = []
    for i, sentence in enumerate(train_df["sentences"]):
        if train_df["classe"].iloc[i] == classe:
            negativas, _, _, palavras_negativas, palavras_positivas, palavras_neutras = contagem_por_polaridade(sentimentos, sentence)
            todas_palavras_negativas.update(palavras_negativas)
            todas_palavras_positivas.update(palavras_positivas)
            todas_palavras_neutras.update(palavras_neutras)

            # Frequência de palavras negativas nas frases
            freq_neg.append(negativas/len(sentence.split()))

    top_negativas = todas_palavras_negativas.most_common()
    top_positivas = todas_palavras_positivas.most_common()
    top_neutras = todas_palavras_neutras.most_common()
    return top_negativas, top_positivas, top_neutras, freq_neg

#print ("\nPalavras negativas mais frequentes na classe Viés: ", palavras_polaridade_top(1)[0])
#print ("\nPalavras negativas mais frequentes na classe Facto: ", palavras_polaridade_top(0)[0])
#print ("\nPalavras negativas mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[0])

#print ("\nPalavras positivas mais frequentes na classe Viés: ", palavras_polaridade_top(1)[1])
#print ("\nPalavras positivas mais frequentes na classe Facto: ", palavras_polaridade_top(0)[1])
#print ("\nPalavras positivas mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[1])

#print ("\nPalavras neutras mais frequentes na classe Viés: ", palavras_polaridade_top(1)[2])
#print ("\nPalavras neutras mais frequentes na classe Facto: ", palavras_polaridade_top(0)[2])
#print ("\nPalavras neutras mais frequentes na classe Citação: ", palavras_polaridade_top(-1)[2])


# -- PALAVRAS EXCLUSIVAS DE CADA CLASSE -- #
# Obtém as palavras mais frequentes por polaridade e classe
so_vies_neg, so_vies_pos, so_vies_neu, _ = palavras_polaridade_top(1)
so_facto_neg, so_facto_pos, so_facto_neu, _ = palavras_polaridade_top(0)
so_citacao_neg, so_citacao_pos, so_citacao_neu, _ = palavras_polaridade_top(-1)

# Extrai só as palavras negativas
palavras_vies_neg = set([p for p, _ in so_vies_neg])
palavras_facto_neg = set([p for p, _ in so_facto_neg])
palavras_citacao_neg = set([p for p, _ in so_citacao_neg])

# Calcula as palavras negativas exclusivas de cada classe
so_vies_neg = palavras_vies_neg - palavras_facto_neg - palavras_citacao_neg
so_facto_neg = palavras_facto_neg - palavras_vies_neg - palavras_citacao_neg
so_citacao_neg = palavras_citacao_neg - palavras_vies_neg - palavras_facto_neg

#print("Palavras só de Viés:", so_vies)
#print("Palavras só de Facto:", so_facto)
#print("Palavras só de Citação:", so_citacao)

# Extrai só as palavras positivas
palavras_vies_pos = set([p for p, _ in so_vies_pos])
palavras_facto_pos = set([p for p, _ in so_facto_pos])
palavras_citacao_pos = set([p for p, _ in so_citacao_pos])

# Calcula as palavras positivas exclusivas de cada classe
so_vies_pos = palavras_vies_pos - palavras_facto_pos - palavras_citacao_pos
so_facto_pos = palavras_facto_pos - palavras_vies_pos - palavras_citacao_pos
so_citacao_pos = palavras_citacao_pos - palavras_vies_pos - palavras_facto_pos

#print("Palavras só de Viés:", so_vies_pos)
#print("Palavras só de Facto:", so_facto_pos)
#print("Palavras só de Citação:", so_citacao_pos)

# Extrai só as palavras neutras
palavras_vies_neu = set([p for p, _ in so_vies_neu])
palavras_facto_neu = set([p for p, _ in so_facto_neu])
palavras_citacao_neu = set([p for p, _ in so_citacao_neu])

# Calcula as palavras neutras exclusivas de cada classe
so_vies_neu = palavras_vies_neu - palavras_facto_neu - palavras_citacao_neu
so_facto_neu = palavras_facto_neu - palavras_vies_neu - palavras_citacao_neu
so_citacao_neu = palavras_citacao_neu - palavras_vies_neu - palavras_facto_neu

#print("Palavras só de Viés:", so_vies_neu)
#print("Palavras só de Facto:", so_facto_neu)
#print("Palavras só de Citação:", so_citacao_neu)

# --- FUNÇÃO PARA CONTAR PALAVRAS EXCLUSIVAS EM CADA FRASE --- #
def contar_palavras_exclusivas(tokens):
    contagens = {
        "so_vies_pos": 0, "so_vies_neg": 0, "so_vies_neu": 0,
        "so_facto_pos": 0, "so_facto_neg": 0, "so_facto_neu": 0,
        "so_citacao_pos": 0, "so_citacao_neg": 0, "so_citacao_neu": 0
    }

    for token in tokens:
        if token in so_vies_pos:
            contagens["so_vies_pos"] = 1
        if token in so_vies_neg:
            contagens["so_vies_neg"] = 1
        if token in so_vies_neu:
            contagens["so_vies_neu"] = 1
        if token in so_facto_pos:
            contagens["so_facto_pos"] = 1
        if token in so_facto_neg:
            contagens["so_facto_neg"] = 1
        if token in so_facto_neu:
            contagens["so_facto_neu"] = 1
        if token in so_citacao_pos:
            contagens["so_citacao_pos"] += 1
        if token in so_citacao_neg:
            contagens["so_citacao_neg"] += 1
        if token in so_citacao_neu:
            contagens["so_citacao_neu"] += 1

    return pd.Series(contagens)

# Aplicar a função de contagem de palavras exclusivas nos Dataframes
train_counts_exclusivas = train_df["tokens"].apply(contar_palavras_exclusivas)
test_counts_exclusivas = test_df["tokens"].apply(contar_palavras_exclusivas)

# -- ADJETIVOS POR CLASSE -- #
def adjetivos_frequentes ():
    top_adjetivos_por_classe = {}
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adjetivos = []

        for doc in subset["doc"]:
            todos_adjetivos.extend([token.text.lower() for token in doc if token.pos_ == "ADJ"])

        fdist = FreqDist(todos_adjetivos)
        top_adjetivos_por_classe[classe] = fdist.most_common(5) # Top 5 adjetivos
    
    return top_adjetivos_por_classe


# -- ADVÉRBIOS POR CLASSE -- #
def adverbios_frequentes ():
    top_adverbios_por_classe = {}

    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_adverbios = []

        for doc in subset["doc"]:
            todos_adverbios.extend([token.text.lower() for token in doc if token.pos_ == "ADV"])

        fdist = FreqDist(todos_adverbios)
        top_adverbios_por_classe[classe] = fdist.most_common(5) # Top 5 advérbios
  
    return top_adverbios_por_classe

top_adjetivos_citacao = adjetivos_frequentes ()[-1]
top_adjetivos_facto = adjetivos_frequentes ()[0]
top_adjetivos_vies = adjetivos_frequentes ()[1]
top_adverbios_citacao = adverbios_frequentes ()[-1]
top_adverbios_facto = adverbios_frequentes ()[0]
top_adverbios_vies = adverbios_frequentes ()[1]

# -- FUNÇÃO PARA CONTAR ADJETIVOS E ADVÉRBIOS FREQUENTES EM CADA FRASE -- #
def contar_adjetivos_adverbios_freq (tokens):
    contagens = {
        "adj_vies" : 0, "adj_facto" : 0, "adj_citacao" : 0, 
        "adv_vies" : 0, "adv_facto" : 0, "adv_citacao" : 0
    }
     
    for token in tokens:
        if token in top_adjetivos_citacao:
            contagens ["adj_citacao"] = 1
        if token in top_adjetivos_facto:
            contagens ["adj_facto"] = 1
        if token in top_adjetivos_vies:
            contagens ["adj_vies"] = 1
        if token in top_adverbios_citacao:
            contagens ["adv_citacao"] = 1
        if token in top_adverbios_facto:
            contagens ["adv_facto"] = 1
        if token in top_adverbios_vies:
            contagens ["adv_vies"] = 1
    
    return pd.Series (contagens)

# Aplicar a função de contagem de adv e adj frequentes nos Dataframes
train_counts_adj_adv = train_df["tokens"].apply(contar_adjetivos_adverbios_freq)
test_counts_adj_adv = test_df["tokens"].apply(contar_adjetivos_adverbios_freq)

# -- PRIMEIRA LETRA MAIÚSCULA DEPOIS DAS ASPAS -- #
def letras_maiusculas (sentence):
    return len(re.findall(r'"[A-ZÁÉÍÓÚÃÕÂÊÔÇ]', sentence))

# -- TEM ASPAS -- #
def contar_aspas(sentence):
    return len(re.findall(r'"', sentence))

# -- TEM PONTO DE INTERROGAÇÃO -- #
def tem_pontuacao (sentence, pontuacao):
    return pontuacao in sentence

# -- SINÓNIMOS DE UMA PALAVRA -- #
def get_sinonimos (word):
    sinonimos_ls = []
    for syn in wordnet.synsets(word, lang = 'por'):
        for lemma in syn.lemmas('por'):
            sinonimos_ls.append(lemma.name())

    sinonimos_ls = list(set(sinonimos_ls)) # Remove palavras duplicadas
    return sinonimos_ls

# -- VERBOS MAIS FREQUENTES POR CLASSE -- #
def verbos_frequentes ():
    top_verbos_por_classe = {} 
    for classe in train_df["classe"].unique():
        subset = train_df[train_df["classe"] == classe]
        todos_verbos = []

        for doc in subset["doc"]:
            todos_verbos.extend([token.lemma_.lower() for token in doc if token.pos_ == "VERB"])

        fdist = FreqDist(todos_verbos)
        top_verbos_por_classe[classe] = fdist.most_common(5) # Top 5 verbos
    
    return top_verbos_por_classe

top_verbos_classe = verbos_frequentes ()

# -- SINÓNIMOS DE UMA LISTA DE VERBOS -- #
def verbos_sinonimos (verbos_frequentes, classe):
    sinonimos = []
    verbos_frequentes = [t[0] for t in verbos_frequentes.get(classe)]
    for verbo in verbos_frequentes:
        if verbo not in sinonimos: 
            sinonimos.append (verbo)

        sinonimos_verbo = get_sinonimos (verbo)

        for sinonimo in sinonimos_verbo:
            if sinonimo not in sinonimos:
                sinonimos.append (sinonimo)
    
    return sinonimos

# Aplicar a função de contagem de verbos frequentes nos Dataframes
top_verbos_citacao = verbos_sinonimos(top_verbos_classe, -1)
top_verbos_facto = verbos_sinonimos(top_verbos_classe, 0)
top_verbos_vies = verbos_sinonimos(top_verbos_classe, 1)

print (top_verbos_classe)
print (top_verbos_citacao)

def contar_verbos (tokens):
    contagens = {
        "verbo_vies" : 0, "verbo_facto" : 0, "verbo_citacao" : 0
    }
     
    for token in tokens:
        if token in top_verbos_citacao:
            contagens ["verbo_citacao"] += 1
        if token in top_verbos_facto:
            contagens ["verbo_facto"] += 1
        if token in top_verbos_vies:
            contagens ["verbo_vies"] += 1
    
    return pd.Series (contagens)

train_counts_verbos = train_df["tokens"].apply(contar_verbos)
test_counts_verbos = test_df["tokens"].apply(contar_verbos)

# Juntar as colunas ao dataframe original
train_df = pd.concat([train_df, train_counts_exclusivas, train_counts_adj_adv, train_counts_verbos], axis = 1)
test_df = pd.concat([test_df, test_counts_exclusivas, test_counts_adj_adv, test_counts_verbos], axis = 1)

# --- FEATURES PARA O MODELO --- #
# Colunas com as contagens
colunas = [
    "so_vies_pos", "so_vies_neg", "so_vies_neu",
    "so_facto_pos", "so_facto_neg", "so_facto_neu",
    "so_citacao_pos", "so_citacao_neg", "so_citacao_neu",
    "adj_vies", "adj_facto", "adj_citacao", "adv_vies",
    "adv_facto", "adv_citacao", "num_aspas", "verbo_vies",
    "verbo_facto", "verbo_citacao"
]

train_df["num_aspas"] = train_df["sentences"].apply(
    lambda s: contar_aspas(s)
)

test_df["num_aspas"] = test_df["sentences"].apply(
    lambda s: contar_aspas(s) 
)

X_train_num = train_df[colunas]
y_train_num = train_df["classe"]

X_test_num = test_df[colunas]
y_test_num = test_df["classe"]

# Para não existirem valores nulos
X_train_num = X_train_num.fillna(0)
X_test_num = X_test_num.fillna(0)
print (y_test_num.shape)
print (X_train_num.shape)

# Definir o número alvo de amostras por classe
target_n = 1200

# Primeiro, aplicar SMOTE para as classes com menos de 1200
smote = SMOTE(sampling_strategy = {-1: target_n, 1: target_n}, random_state = 42)

# Depois, aplicar undersampling para as classes com mais de 1200
under = RandomUnderSampler(sampling_strategy = {0: target_n}, random_state = 42)

# Combinar os dois passos num pipeline
pipeline = Pipeline(steps = [('smote', smote), ('under', under)])

# Aplicar o pipeline
X_train_res, y_train_res = pipeline.fit_resample(X_train_num, y_train_num)

# Verificar o resultado
print("\nDistribuição das classes após SMOTE + undersampling:")
print(pd.Series(y_train_res).value_counts())

## -- MODELO KNN -- ##
def KNN_modelo (n):

    KNN = KNeighborsClassifier (n_neighbors = n) 
    KNN.fit (X_train_res, y_train_res)

    y_pred = KNN.predict (X_test_num)
    return y_pred

y_pred_KNN = KNN_modelo (n = 5)

print("-------- MODELO KNN --------")
print("Accuracy:", accuracy_score(y_test_num, y_pred_KNN))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_KNN, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO KNN 
cm = confusion_matrix(test_df ['classe'], y_pred_KNN)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO KMEANS -- ##
def kmeans (n):
    kmeans = KMeans (n_clusters = n, random_state = 42)
    kmeans.fit (X_train_res)

    y_pred = kmeans.predict (X_test_num)
    return y_pred

y_pred_kmeans = kmeans (n = 3)
y_pred_kmeans = y_pred_kmeans - 1 

print("-------- MODELO K-MEANS --------")
print("Accuracy:", accuracy_score(y_test, y_pred_kmeans))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_kmeans, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO K-MEANS
cm = confusion_matrix(test_df ['classe'], y_pred_kmeans)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo K-Means", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO REGRESSÃO LOGÍSTICA -- ##
def regressao_logistica ():
    log_reg = LogisticRegression (random_state = 42)
    log_reg.fit (X_train_res, y_train_res)

    y_pred = log_reg.predict (X_test_num)
    return y_pred

y_pred_reglog = regressao_logistica ()

print("-------- MODELO REGRESSÃO LOGÍSTICA --------")
print("Accuracy:", accuracy_score(y_test, y_pred_reglog))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_reglog, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO REGRESSÃO LOGÍSTICA
cm = confusion_matrix(test_df ['classe'], y_pred_reglog)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Regressão Logística", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()

## -- MODELO NAIVE BAYES-- ##
def naive_bayes ():
    nb = MultinomialNB()
    nb.fit(X_train_res, y_train_res)

    y_pred = nb.predict(X_test_num)
    return y_pred

y_pred_nb = naive_bayes()

print("-------- MODELO NAIVE BAYES --------")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nRelatório de classificação:")
print(classification_report(test_df ['classe'], y_pred_nb, labels = [-1, 0, 1], target_names = ['Citação (-1)', 'Facto (0)', 'Viés (1)']))
print()

# MATRIZ DE CONFUSÃO DO MODELO NAIVE BAYES
cm = confusion_matrix(test_df ['classe'], y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [-1, 0, 1])
disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
plt.title("Matriz de Confusão do Modelo Naive Bayes", fontsize = 22)
plt.xlabel("Classe Prevista", fontsize = 14)
plt.ylabel("Classe Verdadeira", fontsize = 14)
plt.show()