# Projektplan

## Iteration 1, technischer Durchstich

Stand 25.08.2026:

* Erledigt: lokaler, ignorierter Ablageort `data/test_videos/`
* Erledigt: Video Reader mit Metadaten, Frame Nummer und Zeitstempel
* Erledigt: Start ueber erstes Tor oder manuellen Frame
* Erledigt: interpolierte Tor und Abschnittszeiten mit CSV Export
* Erledigt: relative Durchfahrtsregel fuer bewegte Kamera und bewegtes Tor
* Offen: drei bis fuenf repraesentative Originalvideos ablegen
* Offen: relative Regel und Kamerabewegung auf diesen Videos bewerten
* Offen: manuelle Referenzpunkte fuer die ausgewaehlten Sequenzen erfassen

## Phase 1, Modell
* Datensatz und Labels in Roboflow pruefen, Train/Val/Test Split festlegen
* Modell zur Torerkennung trainieren (Colab, GPU)
* Auf ungesehenen Videos validieren, Metriken dokumentieren (mAP, Precision, Recall je Klasse)

## Phase 2, Tracking und Regel
* Athlet ueber Frames verfolgen (ByteTrack oder BoT SORT)
* Tore der Reihenfolge aus configs/gates_template.yaml zuordnen
* Geometrische Tordurchfahrt Regel: implementiert und getestet in src/timing/geometry.py

## Phase 3, Zeitmessung und Vergleich
* Torzeiten und Abschnittszeiten berechnen (src/timing/lap_time.py)
* Mehrere Fahrten vergleichen (src/comparison/compare_runs.py)
* Ausgabe als Tabelle und Diagramm

## Phase 4, Ausbau
* Markiertes Ergebnisvideo erzeugen
* Einfache Weboberflaeche
* Genauigkeit und Stabilitaet verbessern

## Offene Fragen, vor Phase 1 zu klaeren
* Wie viele gelabelte Bilder liegen aktuell in Roboflow vor, reicht das fuer ein erstes Modell?
* Wird ein Objekterkennungsmodell (Bounding Box je Tor) oder ein
  Segmentierungsmodell benoetigt, angesichts teilweise verdeckter Tore?
* Feste oder bewegte Kamera im ersten Testvideo? Das entscheidet, ob
  eine Bildstabilisierung oder Homographie noetig ist, bevor die
  geometrische Regel auf Bildkoordinaten angewendet werden kann.
* Liegt ein Startsignal im Video vor, das automatisch erkannt werden
  kann, oder wird der Startframe manuell markiert?
