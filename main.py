from models.knn import model

from data_reader import Read

def analyze():

    columns = ['HASTA_NO', 'Cinsiyet', 'Yaş', 'BMI',
       'Sigara kullanımı', 'Antiagregan',
       'NLR', 'Başvuru', 'VİRADS', 'Tm boyutu mm',
       'Tm Sayı', 'Karakteri', 'Yerleşim', 'Mesane boynu tutulumu',
       'Ek sistoskopi bulgu', 'Patoloji', 'Kas var mı', 'Nüks']

    path = "data/data_10092023.xlsx"

    X, y = Read(path,columns).read()

    result_knn = model(X, y)
    result_random = ""
    result_logistic = ""

    analysis = [result_knn, result_random, result_logistic]


    return analysis

if __name__ == "__main__":
    print(analyze())
