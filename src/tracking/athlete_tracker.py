"""
Athletenverfolgung ueber mehrere Frames.

Verantwortlich fuer:
  * Uebergabe der Detektionen an einen Tracking Algorithmus, z.B. ByteTrack
  * Stabile ID Zuweisung ueber kurze Verdeckungen hinweg
  * Ableitung eines einzelnen Bezugspunkts pro Frame,
    zum Beispiel Fusspunktmitte, fuer die geometrische Regel in src/timing/geometry.py

Wird nach Phase 1 implementiert. Offene Frage: reicht ByteTrack
oder wird BoT SORT wegen der bewegten Kamera benoetigt?
"""
