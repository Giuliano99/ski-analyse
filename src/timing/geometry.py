"""
Geometrische Regel zur Erkennung einer Tordurchfahrt.

Ein Tor wird durch zwei Punkte definiert, linke und rechte Torstange,
oder allgemeiner durch eine gedachte Linie zwischen zwei Referenzpunkten.
Der Athlet wird durch einen einzelnen Bezugspunkt repraesentiert,
zum Beispiel die Fusspunktmitte der Bounding Box.

Eine Tordurchfahrt gilt als erkannt, wenn die Verbindungsstrecke
zwischen der Athletenposition im vorherigen und im aktuellen Frame
die Torlinie schneidet. Das wird ueber das Vorzeichen des
Kreuzprodukts bestimmt, ein reiner geometrischer Test ohne
Abhaengigkeit von Modell, Kamera oder Framerate.

Wichtig: Diese Regel liefert nur True oder False fuer "Linie geschnitten".
Ob der Schnittpunkt tatsaechlich zwischen den beiden Torstangen liegt,
prueft segments_intersect zusaetzlich. Nur wenn beides zutrifft,
gilt das Tor als passiert.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def _cross(o: Point, a: Point, b: Point) -> float:
    """Kreuzprodukt der Vektoren OA und OB. Vorzeichen zeigt die Seite an."""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Prueft, ob die Strecke p1 bis p2 (Athletenbewegung)
    die Strecke p3 bis p4 (Torlinie) schneidet.
    Reiner geometrischer Test, keine Rundungstoleranz.
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def gate_crossed(
    prev_athlete_pos: Point,
    curr_athlete_pos: Point,
    gate_left: Point,
    gate_right: Point,
) -> bool:
    """
    Prueft, ob der Athlet zwischen zwei aufeinanderfolgenden Frames
    ein bestimmtes Tor passiert hat.

    prev_athlete_pos, curr_athlete_pos:
        Bezugspunkt des Athleten im vorherigen und aktuellen Frame,
        in Bildkoordinaten oder in einer entzerrten Weltkoordinate.
    gate_left, gate_right:
        Die beiden Referenzpunkte des Tores, zum Beispiel linke
        und rechte Torstange.
    """
    return segments_intersect(prev_athlete_pos, curr_athlete_pos, gate_left, gate_right)
