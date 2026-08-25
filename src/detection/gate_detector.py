"""
Torerkennung.

Verantwortlich fuer:
  * Laden des trainierten Modells (Phase 1 Ergebnis)
  * Inferenz pro Frame, Rueckgabe von Bounding Boxen je erkanntem Tor
  * Zuordnung erkannter Boxen zur Torreihenfolge aus configs/gates_template.yaml

Wird erst implementiert, nachdem in Phase 1 Modell und Exportformat
feststehen. Offene Frage: wird die Torzuordnung ueber raeumliche
Naehe zum vorherigen Frame geloest oder ueber ein zweites Klassifikationsmodell?
"""
