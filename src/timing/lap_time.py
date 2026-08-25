"""
Berechnung von Torzeiten und Abschnittszeiten.

Baut auf den Ergebnissen von geometry.gate_crossed auf. Fuer jedes
erkannte Ereignis werden mindestens folgende Felder gespeichert,
wie im Projektauftrag gefordert:

  gate_id, frame_number, video_timestamp_s, time_since_start_s,
  split_time_s, confidence, uncertain

Der genaue Ueberquerungszeitpunkt zwischen zwei Frames wird per
linearer Interpolation entlang der Bewegungsstrecke geschaetzt,
nicht einfach der Frame mit Ueberquerung genommen. Details werden
in Phase 3 implementiert und mit echten Daten getestet.

Wichtig, siehe Projektauftrag: unsichere oder fehlende Erkennungen
duerfen nicht unbemerkt als korrekt behandelt werden. Jedes Ereignis
muss daher explizit ein uncertain Flag tragen, kein stilles Weglassen.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateCrossingEvent:
    gate_id: int
    frame_number: int
    video_timestamp_s: float
    time_since_start_s: float
    split_time_s: Optional[float]
    confidence: float
    uncertain: bool
    uncertain_reason: Optional[str] = None
