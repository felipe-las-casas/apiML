import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np  

def model(input):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    new_names = ['altura_sépala','largula_sépala','altura_pétala','largura_pétala','tipo_de_iris']
    dataset = pd.read_csv(url, names=new_names, skiprows=0, delimiter=',')

    y = dataset['tipo_de_iris']
    x = dataset.drop(['tipo_de_iris'], axis=1)


    y=pd.get_dummies(y)
    y.sample(9)


    # Gerando conjunto de treino e teste (30% para teste):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) #0.3 data as data test

    # Precisamos converter nosso conjunto de dados para float 32bits, que é o que a MLP recebe:
    x_train = np.array(x_train).astype(np.float32)
    x_test  = np.array(x_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    # Definimos nossa iteração máxima como 5.000 para treinar na época de 5.000 e 
    # alfa como 0,01 para nossa taxa de aprendizado. Definimos verbose como 1 imprimir as saídas 
    # durante o processo de treinamento. Random_state é usado como uma semente aleatória:

    # Importanto a lib de MLP

    # Initialização do modelo:
    Model = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=5000, alpha=0.01, #try change hidden layer
                        solver='sgd', verbose=0,  random_state=233) #try verbode=0 to train with out logging
    #train our model
    h=Model.fit(x_train,y_train)

    #use our model to predict
    y_pred=Model.predict(x_test)

    from sklearn.metrics import mean_squared_error 
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    # Imprime relatório de performance:
    #print(classification_report(y_test,y_pred))
    # Imprime a matriz de confusão (Confusion matrix): 
    #print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    # Imprime a taxa de acerto:  
    #print('accuracy is ',accuracy_score(y_pred,y_test)) 

    # Se quisermos mostrar nosso histórico de loss do modelo, 
    # podemos representar graficamente a propriedade loss_curve_ de nosso modelo treinado:


    plt.plot(h.loss_curve_)
    plt.title('Loss History')
    plt.xlabel('epoch')

    # Lembrando de como codificamos nossas saídas:

    # Iris setosa: 100
    Iris_setosa = np.array( [[1,0,0]] )
    # Iris versicolor: 010 
    Iris_versicolor = np.array( [[0,1,0]] )
    # Iris virginica: 001
    Iris_virginica  = np.array( [[0,0,1]] )

    # Agora podemos fazer previsões com o nosso modelo:
    #entrada_exemplo = np.array([[5.4,3.7,1.5,0.2]])

    y_previsto = Model.predict(input)


    # Caso a saída seja Iris setosa (100):
    if np.array_equal(y_previsto, Iris_setosa) == True:
        return 'Iris setosa'

    # Caso a saída seja Iris versicolor (010):
    if np.array_equal(y_previsto, Iris_versicolor) == True:
        return 'Iris versicolor'

    # Caso a saída seja Iris virginica (001):
    if np.array_equal(y_previsto, Iris_virginica) == True:
        return 'Iris virginica'

