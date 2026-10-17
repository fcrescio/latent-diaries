# Latent Diaries — review critica al 17 ottobre 2026

**Il progetto non dispone ancora di un narratore italiano, fondato su immagini nuove e senza testo in inferenza. Il risultato TB è negativo: il nuovo reader riproduce esempi memorizzati e fallisce su tutte le presentazioni di validazione ascoltate.** La qualità del corpus teacher è migliorata; la capacità del sistema di trasformarla in nuova narrazione non è migliorata in modo dimostrato.

Questa review riguarda l'intero percorso M0–M2 e la successiva esplorazione del feedback audio. Distingue risultati implementativi, giudizi umani e ipotesi. Non avvia training, non cambia backbone, non accede alle 96 immagini riservate e non modifica checkpoint o voti. La revisione sorgente di partenza è `db4cba97e5b31c99d03632e7fdabe54c3759cf8d`. I report precedenti restano come registrazioni storiche; questo documento ne aggiorna l'interpretazione.

**Che cosa dice esattamente l'ultima valutazione.**

Sono stati letti tutti i 12 voti TB, associandoli alla mappatura cieca originale. Il revisore ha dichiarato di essersi concentrato sul secondo frammento, che è quello prodotto dal reader. Non attribuiamo allo student la buona qualità del primo frammento, generato dal teacher testuale.

| Condizione | Clip valutate | Immagini | Note esplicite di incomprensibilità | Voce media | Aderenza media | WAV identici al teacher di training |
|---|---:|---:|---:|---:|---:|---:|
| T0, sola immagine, training | 2 | 2 | 2 | 1 | 1 | 0 |
| T0, sola immagine, validazione | 2 | 2 | 2 | 1 | 1 | 0 |
| T1, immagine + audio, training: storia corretta | 2 | 2 | 0 | 3 | 4 | 2 |
| T1, immagine + audio, training: altro ramo della stessa foto | 2 | 2 | 0 | 3 | 3,5 | 2 |
| T1, immagine + audio, validazione: storia corretta | 2 | 2 | 2 | 1 | 1 | 0 |
| T1, immagine + audio, validazione: altro ramo della stessa foto | 2 | 2 | 2 | 1 | 1 | 0 |

In totale, **8/12 clip hanno una nota esplicita di incomprensibilità**. Tutte le sei presentazioni di validazione, riferite a sole due immagini, hanno quella nota. Le quattro clip senza tale annotazione sono tutte T1 di training e i loro secondi frammenti sono **identici byte per byte ai rispettivi WAV teacher**; la verifica confronta i file effettivi, oltre ai manifest.

L'assenza di una nota negativa non è automaticamente un'approvazione. In questo caso ci sono anche voce 3/5 e aderenza 3–4/5 per quelle quattro clip, ma la loro origine resta memorizzazione verificata. Gli scambi di storia nel training riproducono l'altro target già imparato: non dimostrano invenzione di una continuazione nuova.

Le sei clip di validazione raggiungono tutte il limite di durata. Tuttavia anche due clip di training T0, che terminano prima del limite, sono incomprensibili: il solo controllo EOS/durata non è un filtro di intelligibilità sufficiente. Il punteggio «voce» della gallery non è una scala dedicata all'intelligibilità; qui la conclusione negativa è sostenuta anche dalle note e dal commento esplicito del revisore.

Non stimiamo un tasso di errore sulla popolazione: ci sono un solo revisore, quattro immagini complessive e due immagini di validazione, presentate in più condizioni. La [nuova analisi aggregata](validation/2026-10-17-temporal-human-outcome.json) contiene conteggi e hash, senza note originali, identità o dati privati.

**Che cosa abbiamo realmente costruito.**

| Area | Risultato sostenuto dalle evidenze | Limite che resta |
|---|---|---|
| M0: Qwen ufficiale e codec | Italiano accettato sulla voce fornita; codec e sintesi reale esercitati | È una baseline condizionata sul testo |
| Interfaccia continua | Prefill e trailing path sostituiti correttamente; gradienti, shift, maschere e cache verificati; controlli esatti riproducono il teacher | Un prefisso esatto contiene già l'informazione linguistica della frase |
| M1: controllo numerico | Memorizzazione di frasi distinte; due combinazioni numeriche prespecificate accettate con il sequence reader | Grammatica, lessico e fattori chiusi; un seed e una voce; nessuna immagine |
| Encoder visivo | Feature ufficiali congelate, normalizzate e con provenance verificata | Separazione di categorie non equivale a comprensione dei dettagli necessari a raccontare |
| Prior V8/V9 | Italiano giudicato stabile su sviluppo esposto | Repertorio ristretto, frasi ricorrenti e legame debole con la foto |
| Encoder audio ASFL | Buona corrispondenza semantica su teacher locali; separazione dei significati esercitata | Non è un generatore, una memoria ordinata o un arbitro generale di intelligibilità |
| Corpus a due frammenti | 48 teacher reali; campione TC di otto sequenze giudicato generalmente buono e coerente | È un tetto di qualità testuale, non la qualità del reader |
| Reader temporale TB | Percorso senza testo eseguito e capacità di memorizzazione dimostrata | Nessuna generalizzazione accettata, nemmeno con storia teacher pulita |
| Review e riproducibilità | Gallery, voti, manifest, lock, test, riprese e privacy operativi | Solidità dell'infrastruttura non significa successo del modello |

M0 e M1 conservano la loro accettazione **nel perimetro documentato**. M2 resta aperto. La componente di M4 relativa alla review è disponibile; questo non rende accettata la qualità artistica del narratore. Le comparazioni di M3 e il prototipo finale M5 non sono consegnati come sistema funzionante. `narrate-image` resta intenzionalmente non operativo nella CLI principale, mentre i percorsi di ricerca sono eseguibili tramite script.

La gallery contiene 133 coppie, 132 delle quali hanno un voto salvato, e 85 file audio distinti. Non sono 133 generazioni indipendenti: esistono ripresentazioni, scambi di immagine e composizioni di frammenti. Questi numeri descrivono anche il carico di valutazione chiesto al revisore; non sono una metrica di avanzamento scientifico.

**Come siamo arrivati qui e che cosa ogni fase ha escluso.**

| Fase | Ipotesi provata | Esito e conseguenza |
|---|---|---|
| Inversione M1 | Un piccolo prefisso continuo può controllare il TTS congelato | Sì per frasi note, dopo correzioni e controlli di supervisione/decoding; non è narrazione libera |
| Composizioni numeriche | Il reader può riutilizzare fattori già conosciuti | Il sequence reader passa due combinazioni nuove; la regressione affine non le passa |
| Primo reader di immagini e controllo query/length | Basta mappare il vettore globale nei prefissi delle frasi | Training molto migliore dello sviluppo; intelligibilità non trasferita |
| Target canonici e V4 acustico | Ridurre la variabilità o usare loss audio stabilizza il trasferimento | Alcune frasi risultano italiane, ma ripetitive; nessun miglioramento umano sufficiente |
| Prior V5–V8 | Imparare una distribuzione di prefissi invece di una sola media | Whole-prefix flow e causal flow sono già stati provati; V8 recupera italiano su un repertorio ristretto |
| Adapter, ancore e V9–V15 | Migliorare l'allineamento semantico senza perdere l'italiano | E7: aderenza media 1,75/5 per entrambe le alternative; E8: nessun miglioramento significativo riferito dal revisore |
| V16: target specifici per immagine | La vecchia supervisione era troppo generica | Quattro training WAV identici al teacher, ma sviluppo ancora problematico: una precedente dimostrazione dello stesso divario |
| V17–V20: più target, capacità e LoRA | Più copertura e adattamento del Talker risolvono il limite | 170 target; limite di rango corretto; nessun miglioramento di qualità dimostrato sullo sviluppo |
| V21–V22: distillazione diretta della policy audio | Evitare la regressione dei vettori linguistici | KL bassa su storie gold, divergenza su storie generate; nessun narratore generalizzante |
| V23: repair su storie generate | Ridurre il divario fra training e autoregressione audio | Due iterazioni esercitate; nessuna soluzione dimostrata; il sampling era anche sbilanciato |
| V24: corpus narrativo ampliato | Più contenuti nel curriculum linguistico ripristinano varietà e qualità | 2.927 frasi da 168 gruppi di immagini; l'italiano libero non viene preservato; cambiano anche sottospazio e capacità |
| Oracle ladder OD | Localizzare il primo passaggio che si rompe | Controlli esatti buoni; otto free-prior incomprensibili anche con semantica testuale esatta; la sola visione non spiega il problema |
| ASFL encoder e reranking AF | L'audio rappresentato semanticamente può migliorare la scelta | Encoder promettente; aderenza descrittiva 1,875→2,625/5, ma evocazione e sorpresa non migliorano; nessuna promozione |
| TC e TB | Storie migliori e audio precedente permettono una continuazione | Teacher accettabili; il reader torna alla memorizzazione e fallisce la validazione |

Le ricette non sono confronti tutti simultanei sullo stesso benchmark. Alcune conclusioni negative storiche derivano da diagnostica e ASR, altre da ascolto umano: non le trasformiamo retroattivamente tutte in bocciature umane. I dettagli sono nei [manifest V16–V24](validation/2026-10-16-grounding-reassessment.json), nella [review precedente](REVIEW_RESPONSE_20261016.md), nei [risultati del reranking](validation/audio-feedback-20261016/human-outcome.json) e nel diario [DECISIONS](DECISIONS.md).

**Il collo di bottiglia che emerge è la generazione linguistica continua generalizzante.**

Il percorso effettivo richiede almeno tre competenze diverse: estrarre informazione dalla foto, scegliere e organizzare una nuova frase, realizzarla in suono. Qwen ha dimostrato di pronunciare bene frasi specificate dal testo o dal loro prefisso esatto. Non abbiamo dimostrato che un piccolo reader riesca a produrre, da un embedding semantico, una sequenza continua nuova che Qwen sappia sempre pronunciare come italiano.

Questa distinzione non è solo teorica. Nell'oracle ladder, il prior V24 fallisce già quando riceve la semantica della frase esatta, prima di qualunque adapter di immagini. Nei TB, T1 fallisce con l'audio teacher corretto, quindi **il difetto emerge prima di un loop su audio autoprodotto**. Migliorare la memoria, il reranker o i patch visivi non risolve automaticamente questo passaggio precedente.

È un'inferenza dalle prove locali, non una dimostrazione di impossibilità. Non possiamo concludere che Qwen sia incapace di qualunque generazione textless, che l'embedding visivo sia sufficiente o che un certo numero maggiore di dati garantirebbe la soluzione. Possiamo concludere che la generalizzazione necessaria non è stata ottenuta dalle ricette provate.

Il [report tecnico Qwen3-TTS](https://arxiv.org/abs/2601.15621) presenta la famiglia come sistema di sintesi vocale condizionata sul testo. Non fornisce una garanzia per il nostro uso come narratore aperto da embedding di immagini: tale uso resta una nostra ipotesi di ricerca. La distinzione fra pronunciare contenuto già specificato e apprendere un modello linguistico dal parlato è anche alla base di [Generative Spoken Language Modeling](https://aclanthology.org/2021.tacl-1.79/). Quel lavoro è un riferimento concettuale, non un'alternativa qui verificata per italiano, immagini o dimensioni compatte.

**Perché il confronto TB non era un test adeguato di vantaggio narrativo.**

T0 riceve la stessa immagine per due continuazioni diverse. È deterministico e viene addestrato con errore quadratico a predire entrambe le sequenze di prefissi. A parità di input, quella loss favorisce una media dei target nelle posizioni condivise. Una media di due vettori che corrispondono a parole diverse non è necessariamente una terza frase pronunciabile.

La diagnosi numerica è particolarmente chiara: l'MSE pesata di training di T0, circa 0,472043, coincide con il limite analitico dovuto a questi target contraddittori per una funzione deterministica. Non è un problema risolvibile aumentando gli step di quel modello. La molteplicità delle storie è corretta per la finalità artistica; è l'obiettivo di apprendimento del controllo a non rappresentarla adeguatamente.

T1 riceve un audio diverso per ciascun ramo. Con 4.823.425 parametri e solo 16 continuazioni di training, può associare i vettori di storia ai prefissi esatti. Il successo di training prova una capacità di ricostruzione condizionata; non prova che abbia imparato sintassi, variazione narrativa, progressione o uso utile della foto. Ogni A1 è associato a una sola immagine: il dataset non identifica bene la necessità della memoria visiva permanente. Una variazione numerica del prefisso quando si cambia foto non basta a risolvere questa ambiguità.

Il controllo lessicale offline rende esplicito un altro cambiamento simultaneo. Le 16 continuazioni di training contengono 257 occorrenze di parole e 166 forme distinte; le otto di validazione 123 occorrenze e 92 forme. Di queste ultime, 62 forme, per 64 occorrenze, non compaiono nei target A2 di training. Anche includendo gli A1 di training restano 48 forme nuove. Il conteggio usa parole alfabetiche normalizzate in minuscolo: **non è OOV del tokenizer Qwen** e non prova che sia questa la causa del fallimento. Mostra però che il salto da M1 al pilota mescola nuove immagini, contenuti e forme lessicali, invece di isolare prima la composizione con lessico noto.

Infine, l'assenza di loss audio nel training TB conta: sono stati esercitati gradienti e loss attraverso Qwen come audit, ma i 2.000 aggiornamenti hanno ottimizzato soltanto ricostruzione dei prefissi e lunghezza. Il file [train_temporal_bridge.py](../scripts/train_temporal_bridge.py) lo conferma. Il controllo dei gradienti non può essere descritto come addestramento acustico.

**Registro delle conclusioni: fatti, rischi e ipotesi.**

| Riscontro | Classificazione | Evidenza e limite |
|---|---|---|
| Il reader TB non trasferisce alle immagini di validazione ascoltate | Limite verificato | Sei presentazioni negative su due immagini; non un tasso sulla popolazione |
| I quattro controlli T1 di training riproducono i teacher | Risultato verificato | Hash dei WAV effettivi; è memorizzazione |
| T0 ha target multimodali ma obiettivo deterministico MSE | Difetto metodologico verificato | Codice della loss e limite analitico coincidono col plateau |
| Il loop audio completo risolve il problema | Non dimostrato | Non è mai stato eseguito un rollout A1→A2 interamente generato con qualità accettata |
| T1 usa utilmente l'immagine permanente | Non identificato dal disegno | Storie specifiche per immagine; sensibilità numerica diversa da grounding |
| Il prior continuo è un punto di rottura | Limite verificato per V24 | OD fallisce anche con semantica esatta; non implica fallimento di ogni possibile prior |
| Exposure bias è l'unica causa | Ipotesi non sostenuta come spiegazione unica | Geometry/oracle ancora imperfetti e V23 non risolutivo |
| Il pooling visivo è il problema principale | Ipotesi ancora aperta | La perdita di dettagli è plausibile, ma esistono fallimenti precedenti alla visione |
| Più narrazioni bastano | Non dimostrato | V24 non risolve; parafrasi non aggiungono altrettanti contesti indipendenti |
| Il bug è nel solo decoding greedy | Smentito come spiegazione generale | Campionamento globale stock verificato; TB sampled fallisce comunque |
| I precedenti bug null/provenance spiegano tutto | Non sostenuto | Difetti corretti nella review precedente; fallimenti successivi persistono |
| Audio semanticamente riconoscibile implica capacità di raccontare | Salto logico non valido | Retrieval positivo, generazione temporale negativa |

Non è stato trovato in questa review un nuovo bug di assembly che spieghi i TB. Le prove di parità riducono la plausibilità di quella spiegazione nel percorso esercitato; non certificano l'assenza di ogni bug possibile.

**Dove ho sbagliato nella gestione della ricerca.**

Ho dato troppo peso alla possibilità di fare il passo implementativo successivo e troppo poco alla probabilità che distinguesse ipotesi ancora aperte. La documentazione conservava le cautele, ma non bastava scrivere «non è ancora M2» dopo ogni run: dovevo usare più rigidamente le evidenze negative per scegliere il run seguente.

Il ritorno alla regressione MLP dei prefissi, dopo i fallimenti già osservati, è stato una scelta debole per testare una nuova architettura temporale. Poteva servire come controllo rapido di ricostruzione; non era una base promettente per aspettarsi nuova narrazione. La buona qualità TC rimuoveva un dubbio sul teacher, non sul meccanismo di generalizzazione del reader.

Avrei dovuto esplicitare prima del training che il controllo T0 non poteva rappresentare due esiti con la loss scelta. L'audit successivo lo ha chiarito, ma non recupera il valore causale del confronto. La presenza di molte ablation non rende automaticamente ben identificata una domanda sperimentale.

La gallery TB ha fornito una conferma umana utile, ma il suo valore marginale era limitato da segnali tecnici già negativi e da controlli positivi di training identici ai teacher. Avrei dovuto presentarla come diagnosi finale della ricetta e ridurre il carico di ascolto ai casi decisivi, senza far sembrare la nuova serie un avanzamento del narratore. Il tempo del revisore è una risorsa sperimentale.

Infine, proporre adesso «provare loss acustica», «aggiungere LoRA», «usare un prior generativo/flow» o «fare exposure-bias repair» senza riconoscere gli esperimenti già fatti sarebbe ripetere lo stesso errore. Una variante futura dovrebbe dichiarare quale limite specifico di quei tentativi cambia e quale osservazione la smentirebbe.

**Che cosa salvare e che cosa interrompere.**

Sono da conservare la baseline ufficiale italiana, il codec, l'interfaccia continua verificata, il checkpoint M1 composizionale, il V8/V9 come riferimento ristretto, gli encoder congelati, il corpus TC, i dati e i risultati negativi, la review e la provenance. Nessun checkpoint precedente viene sovrascritto.

**La ricetta temporale TB viene fermata come direzione da scalare.** Il criterio di stop per miglioramento limitato agli esempi memorizzati è soddisfatto. Non aumentiamo frasi, larghezza, step, numero di candidati o lunghezza del loop per compensare questo risultato. L'idea ASFL in generale non è stata confutata: l'encoder audio ha superato controlli necessari, ma il suo uso in un narratore resta non validato. Il ramo di memoria può attendere finché esiste un generatore generalizzante da alimentare.

Non serve un'altra valutazione delle stesse TB per prendere questa decisione. La review richiesta è completa. V25 resta una proposta non avviata e soggetta alla sua autorizzazione specifica; questa review non la riattiva.

**Le alternative, con il loro vero costo logico.**

| Direzione | Stato rispetto al lavoro già fatto | Decisione motivata |
|---|---|---|
| Più MSE, step o larghezza sul reader TB | Ripete un difetto di obiettivo e un risultato di memorizzazione | Scartare per il prossimo ciclo |
| Sostituire MSE con CE/KL, aggiungere LoRA o repair | Principi già esercitati in V4/V16/V19–V23 | Non presentarli come nuova soluzione; servirebbe una differenza discriminante esplicita |
| Flow, prior autoregressivo o mixture | Già esercitati in V5–V8/V24 | Un nome nuovo o più modalità non basta; first gate su frasi nuove prima della visione |
| Più dati/parafrasi | Copertura attuale ridotta; quantità sufficiente ignota | Necessari forse in futuro, ma nessuna base per promettere una soglia risolutiva |
| Feature visive locali/resampler | Contratti parzialmente ispezionati, confronto narrativo non validato | Rinviare finché il generatore fallisce anche con semantica oracle |
| Memoria audio, reranking, più frammenti | Encoder valido; vantaggio artistico e loop non dimostrati | Rinviare l'integrazione; non cura la produzione di italiano nuovo |
| Ricostruire un benchmark composizionale ristretto con reader strutturato | M1 fornisce un precedente positivo, non una soluzione completa | Prosecuzione circoscritta che raccomando dentro il contratto attuale; vedere condizioni sotto |
| Generatore linguistico acustico pretrained diverso | Non valutato per italiano, licenze, risorse o dettaglio d'inferenza | Alternativa architetturale da studiare se l'obiettivo aperto resta prioritario; modifica il vincolo del backbone, non va introdotta tacitamente |
| Visione→testo→Qwen | TC ne mostra il valore come tetto artistico locale | Baseline utile; viola il requisito finale e non sostituisce il progetto textless |

La raccomandazione è **fermare l'accumulo di varianti e decidere esplicitamente che cosa il piccolo reader deve saper generalizzare prima di tornare al narratore aperto**. Il contratto richiede insieme reader piccolo, backbone congelati, italiano nuovo, dipendenza visiva ed espressività. Non abbiamo provato che questo insieme sia raggiungibile con i dati e la rappresentazione attuali; non è corretto gestirlo come un'attività di tuning con esito solo da attendere.

Se manteniamo quel contratto, il prossimo eventuale lavoro dovrebbe essere un singolo benchmark di competenza, non un'altra gallery artistica estesa:

1. Riutilizzare l'idea positiva del sequence reader M1 e fattori semantici continui distinti, mantenendo inizialmente chiusi lessico e strutture. Separare almeno due dettagli variabili; una sola categoria con una coda fissa riprodurrebbe il limite precedente.
2. Congelare prima dello sviluppo combinazioni complete da escludere dal training. Separare il test di composizione di fattori noti dal test di contenuti/forme nuove; il TB li mescolava. Gli input semantici oracle restano controlli offline, mai il prodotto finale.
3. Richiedere italiano autonomo sulle combinazioni escluse, più di un seed prespecificato e nessuna selezione di clip favorevoli. Non riproporre l'oracle ladder già eseguita come se fosse un esperimento nuovo: qui la differenza sarebbe un obiettivo di generalizzazione più ristretto e un reader strutturato, da specificare e motivare rispetto al vecchio query reader.
4. Solo se quel gate passa, verificare che l'immagine permetta di inferire i fattori con split per contesto; poi provare la catena completa con scambi di immagine e null, speaker/seed fissati. Un classificatore grossolano o un input costruito a mano non è accettazione M2.
5. Fermare anche questa strada se passa soltanto il training o se il vantaggio richiede testo, lookup di frasi o codici linguistici recuperati in inferenza. Un successo limitato a una grammatica controllata sarebbe un risultato diagnostico, non il narratore artistico aperto richiesto.

Questo piano è una proposta, non un training lanciato o una promessa di fattibilità. Non fisso un numero di step o una soglia di dati fingendo di conoscerne la sufficienza. Se non accettiamo un passaggio diagnostico ristretto o se fallisce, la decisione successiva è architetturale: riconsiderare il backbone/capacità oppure sospendere l'obiettivo nella forma attuale. Cambiare backbone o introdurre testo richiede una decisione esplicita del proprietario del progetto, non un'eccezione nascosta nel codice.

**Come cambiare il metodo di lavoro.**

Ogni esperimento futuro dovrebbe dichiarare una sola domanda, la differenza rispetto ai tentativi già falliti, la baseline realmente comparabile, il risultato che obbliga a fermarlo e il massimo costo autorizzato. Un nuovo numero di versione non costituisce una nuova ipotesi.

L'unità primaria della valutazione visiva deve essere il contesto/immagine indipendente; varianti, seed e ripresentazioni vanno mantenuti raggruppati. Occorre separare nei report intelligibilità, aderenza, progressione, originalità e robustezza. La gallery attuale non ha una scala dedicata a tutte queste dimensioni: per una futura review pertinente andrà chiarito prima il compito, senza attribuire nuovi punteggi ai voti storici.

Prima di chiedere ascolti estesi, controllare se i casi positivi sono semplicemente WAV già presenti nei teacher, se il controllo di confronto è ben posto e se la diagnostica ha già deciso il criterio di stop. ASR e loss possono motivare un arresto tecnico, mai sostituire l'accettazione umana di qualità. Una baseline stabile ma poco pertinente è un riferimento di regressione, non una soluzione da promuovere per assenza di alternative migliori.

**Fonti, riproduzione e limiti di questa review.**

Sono stati confrontati `DECISIONS.md`, i manifest pubblici dal bootstrap a TB, le precedenti reassessment, i moduli del reader/prior e i relativi trainer, le valutazioni TC/TB/AF/OD, la mappatura privata TB e i WAV di training necessari alla verifica degli hash. La proposta locale dell'utente e la review del 16 ottobre sono state consultate come documenti di ricerca; restano locali e non vengono pubblicate con questo report. I risultati del 17 ottobre hanno una [raccolta aggregata di evidenze](validation/2026-10-17-research-review.json).

La review non è una nuova revisione percettiva dell'audio da parte dell'agente: usa i giudizi reali dell'utente e le misure riproducibili. Non include una ricerca esaustiva di backbone alternativi, una nuova prova di sample complexity o un test fresco del narratore. Le due fonti esterne citate servono a chiarire la distinzione di compiti, non a garantire una soluzione sostitutiva.

La fotografia dei voti e le annotazioni originali restano sotto `artifacts/research-review-20261017/`, ignorato da Git. La nuova analisi dei voti è riproducibile senza modelli o GPU:

```bash
.venv/bin/python scripts/analyze_temporal_bridge_human.py \
  --ratings artifacts/research-review-20261017/ratings-snapshot.json \
  --mapping artifacts/asfl-two-fragment-20261016/bridge-listening/private-mapping.json \
  --teachers artifacts/asfl-two-fragment-20261016/teachers-single-stock/manifest.json \
  --corpus artifacts/asfl-two-fragment-20261016/corpus-v2/corpus.json \
  --output artifacts/research-review-20261017/human-outcome-reproduction.json
```

L'analisi rifiuta voti mancanti/duplicati e condizioni non confrontabili; non inventa valutazioni. Nessuna immagine, voce, trascrizione, nota privata o checkpoint viene caricato su GitHub/HF. Lock e dipendenze non cambiano. Test e controlli sorgente sono registrati nel nuovo ingresso di `DECISIONS.md`.
