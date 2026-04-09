# Descrizione Tecnica del Dataset Multimodale

Questo documento riassume le specifiche tecniche e l'organizzazione strutturale del dataset acquisito tramite lo script `data_collection.py`. Il dataset è volto all'addestramento di un modello per il Multimodal Keyword Spotting (Interazione Multimodale Audio-Video).

## 1. Struttura del Dataset
Il dataset è organizzato in macro-cartelle per ogni singola parola del vocabolario.
- **Vocabolario (10 Parole Chiave italiane):** "Avvia", "Stop", "Sopra", "Sotto", "Sinistra", "Destra", "Apri", "Chiudi", "Sì", "No"
- **Campioni per Parola:** 20 campioni
- **Campioni Totali Registrati:** 200 campioni (10 parole x 20 registrazioni)
- **Durata di ciascun campione:** Esattamente 2 secondi
- **Struttura delle Cartelle:**
  ```text
  dataset/
  └── [Parola_Chiave]/
      ├── audio/ (contiene file .wav)
      └── video/ (contiene file .avi)
  ```
- **Nomenclatura dei File:** `[parola]_[numero_campione]_[timestamp_AAAAMMGG_HHMMSS].[wav|avi]` 
  *(es. `Avvia_001_20260220_103000.wav`)*

## 2. Specifiche Video
I dati visivi sono catturati dalla webcam e processati in tempo reale per estrarre la dinamica labiale.
- **Risoluzione:** 640 x 480 pixel
- **Frame Rate (FPS):** Target a 15 FPS. Per preservare i ritimi reali e gestire lo stuttering hardware, lo script traccia il tempo trascorso calcolando i frame effettivi catturati al secondo (*Actual FPS*).
- **Formato Video / Codec:** `.avi` con codifica `XVID`.
- **Elaborazione Landmark:** Ogni frame subisce un'elaborazione in tempo reale tramite *MediaPipe Face Mesh*, la quale identifica ed evidenzia le labbra disegnando landmark sovrimpressi a schermo e salvati sul video stesso:
  - 20 coordinate per il perimetro delle labbra esterne (verdi).
  - 20 coordinate per il perimetro delle labbra interne (gialle).

## 3. Specifiche Audio
Il parlato è salvato ad altissima qualità per permettere future estrazioni ottimali di features acustiche.
- **Formato File:** Audio non compresso in formato `.wav`
- **Frequenza di Campionamento (Sample Rate):** 44.1 kHz (44100 Hz)
- **Canali:** 1 (Mono)
- **Formato Campione (Bit Depth):** PCM 16-bit (`paInt16`)
- **Buffer di Acquisizione (Chunk):** 1024 frames

## 4. Sincronizzazione Audio-Video
Lo script implementa meccaniche critiche per scongiurare ritardi lip-sync:
- **Flushing dei Buffer (Sync Point):** Immediatamente prima dell'innesco del trigger di ripresa, i buffer hardware (videocamera e input audio) vengono deliberatamente svuotati per garantire lo start a latenza zero di entrambi.
- **Acquisizione in Multithreading:** L'audio viaggia su un thread separato interamente in ascolto, mentre il video raccoglie prima tutti i frame grezzi appoggiandosi unicamente alla RAM, posticipando l'intero carico elaborativo di *MediaPipe* per scongiurare cali di FPS.
- **Allineamento Post-Processing al Millisecondo:** Terminata l'acquisizione, la clip audio riceve un trimming (taglio in eccesso) o padding (aggiunta silenzi in coda) fino a renderla dimensionalmente e temporalmente gemella all'effettiva iterazione dei frame video catturati al netto degli effettivi FPS ricavati.
