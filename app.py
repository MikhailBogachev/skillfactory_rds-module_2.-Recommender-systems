import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse


def nearest_books_nms(itemid, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn


def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('item_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx


def read_files(folder_name='data'):
    """
    Функция для чтения файлов + преобразование к  нижнему регистру
    """
    train = pd.read_csv(folder_name + '/train.csv')
    return train

def nearest_books_nms(itemid, index, n=10):
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn




item_embeddings, nms_idx = load_embeddings()
train = read_files(folder_name='data')


iid = st.text_input('Itemid', '')
iid = int(iid)


output = train[train.itemid == iid]['asin'].loc[train[train.itemid == iid]['asin'].index[0]]
st.write('This product?')
st.write(f'https://www.amazon.com/Miss-Vickies-Jalape%C3%B1o-Flavored-Kettle/dp/{output}')

nbm = nearest_books_nms(iid,nms_idx)[0]
result = train[train.itemid.isin(nbm)][:5]['asin']

'Most simmilar products are: '
for i in range(0, len(result)):
    st.write(f'https://www.amazon.com/Miss-Vickies-Jalape%C3%B1o-Flavored-Kettle/dp/{result.loc[result.index[i]]}')
