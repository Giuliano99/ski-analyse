# Hinweise fuer Codex

## Sicherheit
* Niemals API Schluessel, Tokens oder Zugangsdaten im Code oder in
  Commit Nachrichten speichern. Der Roboflow Key wird lokal aus
  `.env` per `python-dotenv` geladen und in Colab aus Colab Secrets.
* Keine Datensaetze, Videos oder Modellgewichte einchecken. Diese
  Pfade sind bereits in `.gitignore`: data/, datasets/, runs/,
  weights/, *.pt, *.onnx, outputs/, *.mp4, *.mov.
* Vor groesseren Aenderungen einen Git Commit als Sicherungspunkt
  anlegen.

## Vorgehen
* Kleine, ueberpruefbare Schritte. Keine versteckten Annahmen ueber
  Datensatzformat, Kameraposition, Torabfolge oder Modellwahl treffen,
  bei Unklarheit nachfragen statt raten.
* Bestehenden Code vor Aenderungen lesen, nicht blind ueberschreiben.
* Neue Funktionen, insbesondere in src/timing/, mit Unit Tests in
  tests/ absichern. Reine Berechnungslogik soll ohne Video oder Modell
  testbar sein, siehe test_geometry.py als Vorbild.
* Unsichere Erkennungen duerfen nie stillschweigend als sicher gelten,
  siehe GateCrossingEvent.uncertain in src/timing/lap_time.py.

## Struktur
Siehe README.md fuer die Ordnerstruktur. Modulgrenzen einhalten,
Detektion, Tracking, Zeitmessung und Vergleich bleiben getrennt.
