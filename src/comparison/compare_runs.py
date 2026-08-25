"""
Vergleich mehrerer Fahrten.

Erwartet als Eingabe je Fahrt eine Liste von GateCrossingEvent
(siehe src/timing/lap_time.py), typischerweise geladen aus den
exportierten CSV Dateien.

Ausgabe je Tor und Abschnitt:
  * Zeit an jedem Tor
  * Abschnittszeit zwischen zwei Toren
  * Differenz gegenueber einer gewaehlten Referenzfahrt
  * schnellster und langsamster Abschnitt

Wird in Phase 3 implementiert.
"""
