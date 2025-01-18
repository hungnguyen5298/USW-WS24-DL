# Deep Learning-Projekt: Aktienkursvorhersage mit Sentimentdaten und historischen Börsendaten

** Dieses Projekt gehört zum Modul Unternehmenssoftware des Studiengangs Wirtschaftsinformatik an der Hochschule für Technik und Wirtschaft Berlin und wurde im Zeittraum von 01. Okt. 2024 bis 15. Jan. 2025 bearbeitet.**

** Projektstatus: fertig **

______________________
## Inhaltsverzeichnis

Einleitung
    Forschungfrage und Hypothese
    Datenbeschreibung
    Methodik
Implementierung
Ergebnisse
Bekannte Probleme und Einschränkungen
Nutzung
Beitragende
Lizenz

______________________
## Einleitung

1. Fragestellung

Die Sentimentanalyse ist eine Methode, die die Meinungen, Emotionen und Stimmungen aus Texten zu extrahieren. Sie wird häufig im Bereiche wie Kundenservice, Marktanalyse angewendet. Zu den gängigen Ansätze gehören lexikonbasierte Methoden (VADER), sowie maschinelles Lernen (FinBERT).

Die Analyse von Aktienkursen ist eine der größten Herausforderungen in der Finanzforschung. Aktienkurse unterliegen einer Vielzahl von Einflüssen, darunter wirtschaftliche Kennzahlen, politische Ereignisse und Marktsentiment. Traditionelle Prognosemethoden wie statistische Modelle stoßen aufgrund der hohen Volatilität und Komplexität der Märkte oft an ihre Grenzen. In den letzten Jahren hat sich Deep Learning als vielversprechender Ansatz etabliert, da es Muster in großen und komplexen Datensätzen erkennen kann.

Im Finanzmarkt basieren Entscheidungen häufig auf Zahlen und objektiven Daten. Der Einfluss von Stimmungen und Emotionen wird dabei oft unterschätzt, da sie als subjektiv und schwer messbar gelten. Dennoch haben Sentiment-Daten großes Potenzial, da sie Marktreaktionen frühzeitig widerspiegeln können – noch bevor diese sich in Kursbewegungen zeigen. Diese Überlegung bildet die Grundlage für unser Forschungsprojekt.

Aus diesen Gründen haben wir eine Frage gestellt: "Kann die Kombination von Sentimentdaten aus Marktnachrichten mit HLOCV-Daten die Vorhersage von Aktienkursen verbessern?" und eine Hypothese aufgestellt: "Die Kombination von Sentiment-Daten mit Börsendaten verbessert die Genauigkeit bei der Vorhersage von Aktienkursen im Vergleich zur alleinigen Nutzung von Börsendaten."

2. Forschungexperiment

Zur Überprüfung dieser Hypothese wurden drei Experimente durchgeführt:

    - Ein LSTM-Modell, das ausschließlich Börsendaten (HLOCV) nutzt, und dient als Baseline-Modelle.
    - Ein LSTM-Modell, das HLOCV-Daten und Sentiment-Daten aus VADER nutzt.
    - Ein LSTM-Modell, das HLOCV-Daten und Sentiment-Daten aus FinBERT nuzt.

Das Ziel dieser Experimente war es, die Aktienkurse von Apple (AAPL) der nächsten 5 Minuten vorherzusagen. 

## Methodik
1. Projektarchitektur
![projektarchitektur](.image_for_documentation/project_architektur.png)


2. Datenquellen und -beschreibung
In diesem Projekt werden 2 Arten von Inputdaten verwendet:

|               |                                   historische Börsendaten                                    |                                                                                                           Sentimentdaten                                                                                                            |
|:--------------|:--------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Eigenschaften |                                      stetige Zeitreihen                                      |                                                                                                       Ereignisbasierte Daten                                                                                                        |
| Zeitraum      |                                 04.Dez.2024 - 06. Jan. 2025                                  |                                                                                                     04.Dez.2024 - 06. Jan. 2025                                                                                                     |
| Frequenz      | 5-Minuten-Intervall, täglich von 09:30 bis 16:00 Uhr<br/>Außer an Feiertagen und Wochenenden |                                                                                                        keine feste Frequenz                                                                                                         |
| Inhalt        |                              High, Low, Open, Close und Volume                               | - Vorverarbeitung: Texte, die sich auf Apple Inc. (AAPL) und deren Produkte beziehen, darunter Nachrichten, Artikel, Social-Media-Posts und Kommentare.<br/>- Nachverarbeitung: Sentiment-Scores, extrahiert mit VADER und FinBERT. |
| Quelle        |                                      Yahoo Finance API                                       | Daten stammen aus verschiedenen Plattformen und Publikationen, einschließlich Reddit, Yahoo News, Google News sowie anderer relevanter Publisher.                                                                                                                                                                                                                                    |


3. Datenaufbereitung - Vorstellung von Datenpipeline
4. Verwendete Modelle & Modellkonfiguration

## Ergebnisse
1. Evaluierungsmetriken
2. Vergleich der Ergebnisse

## Bekannte Probleme & Einschränkungen

## Weitere Details
1. Nutzung
2. Beitragende
3. Kontakt
