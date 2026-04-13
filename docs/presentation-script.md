# Script di presentazione — Audio-Visual Speech Recognition
> Tono: discorsivo/naturale | Lingua: italiano | Formato: bullet points parlati

---

## Slide 1 — Titolo

- **Presentati** → "Buongiorno, sono Diego Barreto, matricola 1901707. Presento il progetto di Multimodal Interaction: Audio-Visual Speech Recognition, o AVSR."
- **Dai contesto** → "L'idea di base è combinare audio e video per costruire un sistema di riconoscimento vocale più robusto di quanto si possa fare con il solo audio."

> *Transizione:* "Partiamo dal perché questo problema è interessante."

---

## Slide 2 — Motivazione

- **Problema dell'ASR classico** → "I sistemi di riconoscimento vocale automatico funzionano bene in condizioni controllate, ma degradano rapidamente appena aggiungi del rumore reale."
- **Cocktail party effect** → "Il caso classico è il cosiddetto cocktail party effect: quando ci sono più voci sovrapposte, i formanti del parlato vengono mascherati e il modello fatica a distinguere le parole."
- **McGurk effect — la chiave** → "Il punto interessante è che esiste un'altra fonte di informazione: i movimenti delle labbra. La modalità visiva è invariante al rumore acustico — le labbra si muovono allo stesso modo sia in silenzio che in mezzo al caos."
- **Goal del progetto** → "Quindi l'obiettivo è classificare 10 comandi italiani parlati in 6 condizioni di degrado diverse, usando un'architettura che fonde audio e video."

> *Transizione:* "Prima di entrare nei dettagli tecnici, vi mostro le tre cose originali che abbiamo fatto."

---

## Slide 3 — Contributi del progetto

- **Primo contributo** → "Abbiamo progettato un'architettura di Early Fusion multimodale: un branch audio basato su LSTM e un branch video basato su BiLSTM, fusi a livello di feature."
- **Secondo contributo** → "Abbiamo identificato e risolto due bias sperimentali che gonfiavano artificialmente le performance: il Data Leakage e il Padding Leakage — ne parleremo tra poco."
- **Terzo contributo** → "Infine, abbiamo costruito una Ablation Matrix a 6 condizioni per validare empiricamente l'ortogonalità delle modalità, la sinergia multimodale e l'assenza di negative transfer."

> *Transizione:* "Iniziamo dai dati."

---

## Slide 4 — Dataset

- **I 10 comandi** → "Il dataset è composto da 10 comandi italiani parlati: avvia, stop, sopra, sotto, sinistra, destra, apri, chiudi, sì, no. Ogni campione è una coppia sincronizzata di file WAV e MP4."
- **Setting controllato** → "È un singolo speaker in ambiente controllato, quindi ci concentriamo sulla robustezza al rumore artificiale piuttosto che alla variabilità del parlante."
- **Augmentation pipeline** → "Per simulare condizioni reali, abbiamo applicato quattro tipi di degradazione: Babble Noise a +5 dB e -15 dB di SNR per l'audio, Spatial Jitter e Static Tilt per il video, più una combinazione audio-video."

> *Transizione:* "Vediamo come estraiamo le feature da questi dati, partendo dall'audio."

---

## Slide 5 — Estrazione feature audio

- **Pipeline di base** → "Il segnale audio viene ricampionato a 16 kHz. Da ogni frame estraiamo 13 MFCC, e fissiamo la lunghezza della sequenza a 30 frame — il tensore di input è quindi (30, 13)."
- **Perché gli MFCC** → "La scelta degli MFCC non è casuale: la DCT decorrelates l'inviluppo spettrale, rendendo le feature indipendenti tra loro. E cosa più importante, gli MFCC sono una rappresentazione speaker-independent — il modello impara la fonetica, non la voce specifica del parlante."
- **Contrasto con lo spettrogramma grezzo** → "Uno spettrogramma grezzo porterebbe troppo rumore ridondante — gli MFCC isolano proprio i formanti fonetici che ci interessano."

> *Transizione:* "Per il video il discorso è più interessante, perché la prima versione naïve aveva un problema serio."

---

## Slide 6 — Estrazione feature video

- **MediaPipe Face Mesh** → "Usiamo MediaPipe Face Mesh per rilevare i landmark facciali 3D. Da questi estraiamo 3 misure geometriche per frame: apertura interna, apertura esterna e larghezza delle labbra."
- **Il problema** → "Il problema è che queste misure statiche causano gradient stagnation durante il training: il modello non riesce ad imparare perché le feature non variano abbastanza."
- **La soluzione: derivate cinematiche** → "La soluzione è aggiungere le derivate cinematiche: Δ, cioè la velocità, e ΔΔ, cioè l'accelerazione. Passiamo da 3 a 9 feature per frame — il tensore diventa (30, 9)."
- **Perché funziona** → "In pratica non stiamo più descrivendo la posizione delle labbra, ma il loro movimento. E il movimento è il vero segnale discriminativo per il riconoscimento dei comandi."

> *Transizione:* "Prima di parlare dell'architettura, dobbiamo affrontare due bias che avrebbero reso i risultati completamente inutili."

---

## Slide 7 — Bias #1: Data Leakage

- **Il problema** → "Con uno split casuale, le varianti augmentate dello stesso campione finiscono sia nel training set che nel test set. Il modello vede di fatto le stesse registrazioni durante il training e la valutazione."
- **L'effetto** → "Il risultato è drammatico: accuracy al 100% anche a -15 dB di rumore. Non perché il modello sia bravo, ma perché ha memorizzato i campioni specifici del speaker."
- **La soluzione: Stratified Group K-Fold** → "La soluzione è un approccio a due livelli. Prima usiamo un GroupKFold con n=7 per ritagliare un test set fisso — il gruppo è il nome del file base, così tutte le varianti augmentate di uno stesso campione restano nella stessa partizione. Poi un secondo GroupKFold con n=5 divide il resto in train e validation, con stratificazione per bilanciare le classi."

> *Transizione:* "Il secondo bias era meno ovvio, ma altrettanto invalidante."

---

## Slide 8 — Bias #2: Padding Leakage

- **Il problema** → "Per portare tutte le sequenze a 30 frame, la soluzione più semplice è aggiungere zero-padding alla fine. Ma c'è un effetto collaterale: l'LSTM impara a contare i frame a zero per inferire la durata del comando."
- **Perché è un bias** → "Parole lunghe come 'sinistra' hanno pochi zero-frame, parole corte come 'sì' ne hanno molti. Il modello bypassa completamente l'apprendimento fonetico — distingue le parole dalla loro durata, non dal loro suono."
- **La soluzione: Continuous Noise Canvas** → "La soluzione è il Continuous Noise Canvas: generiamo un canvas di rumore lungo 30 frame, poi sovrapponiamo il segnale vocale a un offset randomizzato. In questo modo ogni frame contiene dati non-zero e la durata diventa invisibile al modello."

> *Transizione:* "Con questi bias eliminati, passiamo alle architetture."

---

## Slide 9 — Architettura: Audio-Only LSTM

- **Stack** → "Il modello audio è semplice: BatchNorm → LSTM con 64 unità → Dropout a 0.4 → Softmax su 10 classi."
- **Perché unidirezionale** → "La scelta di un LSTM unidirezionale è deliberata: i formanti del parlato si sviluppano causalmente nel tempo. Ogni fonema dipende da quello precedente, non da quello successivo. Un modello bidirezionale non porterebbe vantaggio e aggiungerebbe complessità inutile."
- **BatchNorm** → "Il BatchNorm prima dell'LSTM normalizza gli MFCC online durante il training, evitando di dover fare normalizzazione offline sui dati."

> *Transizione:* "Per il video la storia è diversa, e la scelta di architettura cambia."

---

## Slide 10 — Architettura: Video-Only BiLSTM

- **Stack** → "Il modello video ha uno strato in più: Masking → BatchNorm → BiLSTM 64 unità → Dropout 0.5 → Softmax."
- **Perché bidirezionale** → "I movimenti delle labbra hanno un fenomeno che si chiama coarticolazione: la bocca inizia a prepararsi per il fonema successivo mentre sta ancora producendo quello corrente. Per catturare questo, il modello ha bisogno del contesto futuro — e per questo usiamo un BiLSTM."
- **Il Masking layer** → "Il Masking layer dice all'LSTM di ignorare i frame a zero. Questo è necessario qui perché nel video non abbiamo applicato il Noise Canvas — i landmark mediaPipe a zero segnalano frame non validi, non rumore."

> *Transizione:* "Il modello di fusione combina entrambi i branch."

---

## Slide 11 — Early Fusion Multimodale

- **Struttura** → "L'architettura di fusione è semplice: i due branch processano audio e video in parallelo, poi i loro hidden state vengono concatenati in un vettore a 128 dimensioni."
- **Il layer condiviso** → "Sopra la concatenazione c'è un Dense da 64 con ReLU: questo layer impara le correlazioni cross-modali. Per esempio, il silenzio acustico correlato alla chiusura delle labbra. È questo layer che fa il lavoro di 'fusione' vera."
- **Perché Early Fusion** → "Chiamiamo questo Early Fusion perché la fusione avviene a livello di feature, non di decisione. Questo permette al modello di integrare le due modalità prima di prendere una decisione finale."

> *Transizione:* "Vediamo ora se questa architettura funziona davvero."

---

## Slide 12 — Ablation Matrix

- **Lettura della tabella** → "In questa tabella ogni riga è un modello, ogni colonna è una condizione di degrado. Le condizioni sono: Clean, Babble a +5 dB, Babble a -15 dB, Jitter, Tilt, e la combinazione audio-video leggera."
- **Il dato più importante** → "Guardando l'accuracy complessiva: la Fusion ottiene 96.55%, l'Audio 90.23%, il Video 77.01%. La fusione batte entrambe le singole modalità."
- **Il caso critico** → "Il caso più interessante è Audio Heavy, cioè -15 dB di Babble. L'Audio-only crolla a 44.83% — quasi il caso del caso. La Fusion invece tiene a 86.21%. Il video salva il modello."
- **Simmetria** → "Succede anche il contrario: con Video Light, il Video-only scende a 41.38%, ma la Fusion recupera a 100%. In questo caso è l'audio a fare da ancoraggio."

> *Transizione:* "Analizziamo più nel dettaglio questi due recovery."

---

## Slide 13 — Sinergia Multimodale e Dynamic Gating

- **Recovery audio** → "Sotto Babble a -15 dB, passiamo da 44.83% a 86.21%, un recupero di 41 punti percentuali. Le feature visive agiscono da ancora quando l'audio è completamente degradato."
- **Recovery video** → "Sotto Spatial Jitter, passiamo da 41.38% a 100%, un recupero di quasi 59 punti. In questo caso il modello ignora i dati cinematici corrotti e si affida interamente all'audio pulito."
- **Comportamento emergente** → "Questo comportamento non è stato programmato esplicitamente: è il Dense condiviso che ha imparato a pesare le modalità in base alla loro affidabilità. È una forma di gating implicito che emerge dal training."
- **Il caso A+V Light** → "Quando degrado sia audio che video contemporaneamente, la fusion tiene a 96.55%. Nessun negative transfer — il modello non peggiora rispetto alla singola modalità migliore."

> *Transizione:* "Le learning curves confermano questo quadro."

---

## Slide 14 — Learning Curves

- **Audio** → "L'LSTM audio converge in modo stabile e rapido. La validation accuracy segue da vicino quella di training — nessun overfitting significativo."
- **Video** → "Il BiLSTM video è più lento e rumoroso. Ha senso: le feature cinematiche portano meno segnale discriminativo degli MFCC, quindi il modello fa più fatica."
- **Fusion** → "Il modello di fusione converge più velocemente di entrambi. Le due modalità si complementano durante il training — il segnale complessivo è più informativo e il modello trova la soluzione più in fretta."

> *Transizione:* "Le confusion matrix ci dicono dove ogni modello sbaglia."

---

## Slide 15 — Confusion Matrix

- **Audio** → "Per l'LSTM audio, gli errori si concentrano quasi tutti nella condizione Audio Heavy a -15 dB. Nelle altre condizioni la diagonale è praticamente pulita."
- **Video** → "Per il BiLSTM video, gli errori si concentrano in Video Light, la condizione con Spatial Jitter. Il jitter distrugge le derivate cinematiche Δ e ΔΔ — che sono proprio le feature più discriminative."
- **Fusion** → "La confusion matrix della Fusion ha la diagonale più pulita di tutte. L'unica condizione dove persistono errori residui è ancora Audio Heavy, che è il caso di degrado più estremo nel dataset."

> *Transizione:* "Concludiamo con i takeaway principali e dove si può andare da qui."

---

## Slide 16 — Conclusioni e Lavori Futuri

- **Risultato principale** → "L'architettura di Early Fusion raggiunge il 96.55% di accuracy complessiva, contro il 90.23% dell'Audio-only e il 77.01% del Video-only."
- **Insight chiave** → "La cosa più interessante non è il numero assoluto, ma la robustezza: sotto rumore catastrofico a -15 dB, la fusione recupera oltre 40 punti percentuali rispetto all'audio da solo. Le due modalità sono ortogonali — quando una fallisce, l'altra compensa."
- **Limitazioni** → "Il lavoro ha due limitazioni principali: è un singolo speaker in ambiente controllato, e la concatenazione semplice dell'Early Fusion non può fare gating dinamico per-frame — pesa le modalità in modo uguale per tutta la sequenza."
- **Lavori futuri** → "Le direzioni più interessanti sono: Cross-Modal Attention e architetture Transformer per un gating dinamico vero per-frame, espansione a più speaker e a vocabolario continuo, e il deployment su edge device tramite quantizzazione del modello."
- **Chiudi** → "Grazie. Sono disponibile per domande."
