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

## Datensatz und Modellgewichte

Der gelabelte Datensatz liegt in Roboflow, nicht in diesem Repository.
Trainingsgewichte werden nach dem Colab Training in Google Drive oder
als GitHub Release abgelegt, ebenfalls nicht direkt eingecheckt.

## Vorgehen

Siehe `docs/projektplan.md` fuer die vollstaendige Phasenplanung.
