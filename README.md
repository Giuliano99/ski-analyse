# Ski Renntraining Analyse

Automatische Analyse von Ski Renntrainingsvideos. Erkennung von Toren
und Athlet, Bestimmung der Tordurchfahrten und Zeitmessung, Vergleich
mehrerer Fahrten.

## Status

Projektaufbau, Phase 1 (Datensatz und Modelltraining) noch offen.
Die geometrische Regel zur Tordurchfahrt (`src/timing/geometry.py`)
ist implementiert und getestet, unabhaengig vom Modell nutzbar.

## Projektstruktur

```
notebooks/       Colab Trainingsnotebook
src/io/          Video einlesen
src/detection/   Torerkennung, Modellinferenz
src/tracking/    Athletenverfolgung ueber Frames
src/timing/      Tordurchfahrt Regel, Zeitberechnung
src/comparison/  Vergleich mehrerer Fahrten
src/visualization/  Tabellen, Diagramme, markiertes Video
configs/         Trainingskonfiguration, Torreihenfolge je Strecke
tests/           Unit Tests
outputs/         Generierte CSV und Analysevideos, nicht versioniert
```

## Setup lokal

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env mit echtem ROBOFLOW_API_KEY befuellen, wird nie eingecheckt
```

## Tests ausfuehren

```
python -m pytest tests/ -v
```

## Testvideos

Lokale Testvideos gehoeren nach `data/test_videos/`. Der Ordnerinhalt
ist ignoriert und darf nicht eingecheckt werden. Die Zeitmessung kann
das erste Tor als Zeitnullpunkt verwenden oder einen manuellen
Startframe aus der Laufkonfiguration lesen.

Eine Laufkonfiguration wird aus der Vorlage erstellt:

```
cp configs/gates_template.yaml configs/run_test.yaml
```

`timing.start_mode` ist entweder `first_gate` oder `manual_frame`.
Beim manuellen Modus muss `timing.manual_start_frame` gesetzt sein.

## Manuelle Referenzannotation

Zuerst mit `run3.mp4` starten:

```
python -m src.annotation.manual_annotator data/test_videos/run3.mp4 --athlete "Athlet in Rot"
```

Bedienung im Videofenster:

* `A` und `D`: ein Frame zurueck oder vor
* `J` und `L`: fuenf Frames zurueck oder vor
* `U` und `O`: 30 Frames zurueck oder vor
* `N`: neue Tordurchfahrt beginnen
* Je Frame: Athlet, linken Torpunkt und rechten Torpunkt anklicken
* Rechtsklick: letzten Punkt entfernen
* `Z`: aktuelle Markierung abbrechen
* `X`: zuletzt gespeichertes Tor entfernen
* `Q`: speichern und beenden

Die beiden Torpunkte definieren die gedachte Zeitmesslinie. Bei einem
Riesenslalom sind das normalerweise die relevanten inneren Stangen der
zusammengehoerigen Tore, nicht automatisch die zwei Stangen desselben
Banners. Als Athletenpunkt wird die Mitte zwischen den Skiern auf
Bodenhoehe verwendet. Diese Definition muss fuer alle Videos gleich sein.

Nach der Annotation werden die Referenzzeiten erzeugt:

```
python -m src.annotation.export_manual_times data/manual_annotations/run3.json
```

Die CSV liegt anschließend unter `outputs/run3_times.csv`.

Wenn die beiden markierten Frames keinen eindeutigen geometrischen
Seitenwechsel enthalten, fordert das Werkzeug standardmaessig einen
anderen zweiten Frame an. Fuer belastbare Referenzen sollten die Frames
unmittelbar vor und nach dem Seitenwechsel liegen. Nur fuer einen
bewussten Entwurf duerfen solche Markierungen mit `--allow-uncertain`
gespeichert werden; der Export verwendet dann die Mitte des Intervalls
und setzt `uncertain=true`.

## Datensatz und Modellgewichte

Der gelabelte Datensatz liegt in Roboflow, nicht in diesem Repository.
Trainingsgewichte werden nach dem Colab Training in Google Drive oder
als GitHub Release abgelegt, ebenfalls nicht direkt eingecheckt.

Das aktuelle GS-Modell ist Roboflow Dataset-Version 8 mit YOLOv11n.
Fuer einen sparsamen Smoke-Test wird standardmaessig ein Frame pro
Sekunde ueber die Hosted API ausgewertet:

```
python -m src.detection.analyze_video data/test_videos/run3.mp4 --max-samples 10
```

Die JSON-Ergebnisse und markierten Vorschaubilder liegen danach unter
`outputs/gate_detection/run3/`. Der API-Key wird ausschliesslich aus der
Umgebungsvariable `ROBOFLOW_API_KEY` gelesen.

## Athlet fuer das Tracking auswaehlen

Fuer den MVP wird der Zielathlet einmal pro Video manuell initialisiert:

```
python -m src.tracking.select_athlete data/test_videos/run3.mp4
```

Mit `A/D`, `J/L` und `U/O` zum ersten Frame navigieren, in dem der
Athlet vollstaendig und gut sichtbar ist. Dann `B` druecken, eine enge
Box um den gesamten Athleten ziehen und mit Enter oder Leertaste
bestaetigen. Die Auswahl wird unter
`data/athlete_selections/run3.json` gespeichert.

Danach wird die Spur lokal mit CSRT und einer YOLO-Personenerkennung
erzeugt. Das Video wird dabei nicht hochgeladen:

```
python -m src.tracking.track_athlete data/test_videos/run3.mp4
```

Die Spur und sekündlichen Vorschaubilder liegen unter
`outputs/athlete_tracking/run3/`. Jeder Frame enthaelt zusaetzlich
`source`, `confidence` und `uncertain`. Frames mit `uncertain: true`
duerfen nicht ungeprueft fuer die Zeitmessung verwendet werden.

Zum isolierten Vergleich stehen `--method csrt` und `--method mil`
zur Verfuegung. Der erste Aufruf des Hybridtrackers benoetigt die lokal
gespeicherte Datei `yolo11n.pt`; Modellgewichte sind per `.gitignore`
vom Repository ausgeschlossen.

## Athletenspur mit manuellen Torzeiten vergleichen

Die automatische Athletenspur kann im lokalen Suchfenster gegen die
manuell markierten, bewegten Torlinien ausgewertet werden:

```
python -m src.comparison.compare_tracking_to_manual data/manual_annotations/run3.json outputs/athlete_tracking/run3/tracking.json
```

Der Bericht liegt unter `outputs/timing_comparison/run3.csv`. Ein Rohfehler
wird auch fuer unsichere Referenzen ausgegeben, aber nur Zeilen mit
`uncertain: false` gelten als belastbarer Zeitvergleich.

Die bestaetigten Durchfahrtsframes fuer `run3` stehen unabhaengig von den
Punktannotationen in `configs/references/run3.yaml`. Automatische Ergebnisse
gelten dort bei einer Abweichung von hoechstens zwei Frames als korrekt.

## Vorgehen

Siehe `docs/projektplan.md` fuer die vollstaendige Phasenplanung.
