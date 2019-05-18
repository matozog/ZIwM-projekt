# ZIwM-projekt

### Temat: Komputerowe wspomaganie diagnozowania nowotworów piersi z wykorzystaniem algorytmów minimalno-odległościowych

## Autorzy

* **Mateusz Ożóg** 
* **Grzegorz Milaszkiewicz** 

## Wymagania

Do uruchomienia projektu potrzebne nam są:
* python w wersji 3,
* biblioteka sklearn.

Interpreter python można pobrać z oficjalnej strony producenta oprogramowania. Do zainstalowania biblioteka sklearn należy użyć komendy: 

> pip install scikit-learn

Jeżeli nie posiadasz narzędzia pip możesz je ściągnąć korzystającz instrukcji znajdującej się w tutaj: https://pip.pypa.io/en/stable/installing/.

## Uruchomienie

Po ściągnieciu repozytorium w głównym folderze (tam gdzie znajduje się plik main.py) należy uruchomić program w wierszu poleceń przy pomocy komendy:

> python main.py

W konsoli wyświetlą się przykładowe wyniki wraz z krótkim opisem.

## Obsługa

Aby zmieniać poszczególne parametry projektu, należy edytować plik main.py. Znajdziemy w nim wszystkie niezbędne informacje. Dodatkowo zostały tam umieszczone przykłady uruchomienia poszczególnych algorytmów np.:

```python    

    #others simple examples

    # features = [7, 17, 4, 5, 12] (index of used featuers)
    # accuracy_knn_alg, matrix = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], 1, algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])
    # accuracy_nm_alg, matrix = nm_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])

    # print(accuracy_knn_alg)
    # print(accuracy_nm_alg)  
```
    
Jest również możliwość uruchomienia testów zwracających wyniki 5 powtórzeń 2-krotnej walidacji krzyżowej. Należy odkomentować poniższą linię kodu:

```python
    # function to create researches (save results to .csv file)
    # createResearches(patients, algorithms, ksData, algParameters)  # function using to create researches
```
Przed uruchomieniem programu należy stworzyć folder results w folderze głównym (tam gdzie plik main.py). Funkcja ta z racji przeliczania dość dużej ilości danych wykonuje się chwilę. Aby zmniejszyć ilość powtórzeń 2-krotnej walidacji krzyżowej należy w pliku src/utils/researches.py zmienić wartość zmiennej *amountOfLoops*.

``` python
  # parameters to cross validation
amountOfLoops = 5

def createTeachingAndTestSets(patients):
    teachingSet = []
    testSet = []
 .........................
```
