"""Authored training-only Italian combinations; no private media descriptions."""

OPENINGS = {
    'water': ["L'acqua conservava il colore del cielo", 'Le onde cancellavano le impronte sulla riva',
              'Il fiume portava via piccoli riflessi', 'Sulla superficie tremava una luce chiara',
              'Il mare respirava contro le pietre', 'La riva seguiva il passo delle onde'],
    'flowers': ['I petali trattenevano una goccia di luce', 'Un fiore si piegava verso il sentiero',
                'Il giardino accendeva piccole macchie di colore', 'Tra i fiori passava un vento leggero',
                'Le corolle nascondevano ombre quasi azzurre', 'Una rosa custodiva il calore del pomeriggio'],
    'trees': ['Gli alberi dividevano la luce in frammenti', 'Le foglie tremavano sopra il sentiero',
              'Il bosco raccoglieva gli ultimi rumori', 'Un ramo disegnava una curva nel cielo',
              'Le ombre degli alberi attraversavano la strada', 'Tra i tronchi filtrava una luce sottile'],
    'mountains': ['Le montagne segnavano il confine del giorno', 'La roccia tratteneva il calore del sole',
                  'Le cime si perdevano dentro le nuvole', 'Il sentiero saliva tra pietre chiare',
                  'Sui monti passava una luce fredda', 'Il profilo delle rocce sembrava una scrittura'],
    'snow': ['La neve copriva ogni traccia del sentiero', 'Il bianco rendeva lontani tutti i rumori',
             'Sui rami brillavano piccoli cristalli di ghiaccio', 'La neve restituiva una luce quasi azzurra',
             'Il pendio conservava poche impronte leggere', 'Una distesa bianca cancellava i confini'],
    'buildings': ['Le finestre raccoglievano gli ultimi riflessi', 'La strada passava tra facciate silenziose',
                  'Un muro conservava le crepe degli anni', 'I tetti dividevano il cielo in pezzi',
                  'Tra le case resisteva una luce calda', 'Il palazzo allungava la propria ombra'],
    'vehicles': ['Le ruote sembravano pronte a ripartire', 'Un veicolo aspettava lungo la strada',
                 'Sul metallo scivolava un riflesso del cielo', 'La carrozzeria tratteneva il colore del giorno',
                 'Il viaggio cominciava da quelle ruote ferme', 'Tra i veicoli rimaneva una striscia di luce'],
    'animals': ["L'animale seguiva un rumore appena percepibile", 'Un piccolo movimento interrompeva la quiete',
                'Lo sguardo del gatto inseguiva una luce', 'Il cane ascoltava qualcosa oltre il bordo',
                'Una creatura si fermava vicino al sentiero', 'Il pelo raccoglieva i riflessi del sole'],
    'food': ['La crosta del pane custodiva un calore', 'Sul piatto restavano colori ancora vivi',
             'La frutta raccoglieva la luce della finestra', 'Il cibo portava con sé una stagione',
             'La tavola sembrava pronta per un incontro', 'Un riflesso dorato attraversava il piatto'],
    'interior': ['La stanza raccoglieva una luce obliqua', 'Sui mobili riposava il calore del giorno',
                 'La finestra ritagliava un quadrato luminoso', 'Una sedia aspettava accanto alla parete',
                 'Le ombre attraversavano lentamente il pavimento', 'Ogni oggetto occupava un piccolo silenzio'],
}

CONTINUATIONS = [
    'il giorno cambiava lentamente colore.', 'una pausa rendeva tutto più vicino.',
    'qualcosa invitava a fermarsi ancora.', 'il tempo sembrava perdere la strada.',
    'un piccolo dettaglio chiedeva attenzione.', 'la distanza diventava improvvisamente leggera.',
    'quel momento aveva un peso diverso.', 'rimaneva soltanto il desiderio di guardare.',
    'la luce spostava ogni certezza.', 'nessuno aveva fretta di ripartire.',
    'il pomeriggio scivolava verso la sera.', 'una presenza sembrava vicina e invisibile.',
    'lo sguardo trovava un posto tranquillo.', 'il mondo respirava per un istante.',
    'qualcosa cominciava senza fare rumore.', 'il ricordo prendeva una forma nuova.',
    'un pensiero tornava da molto lontano.', 'la quiete aveva un colore preciso.',
    'il vento portava una domanda diversa.', 'ogni distanza sembrava ancora possibile.',
    'il passo rallentava quasi senza motivo.', 'un movimento cambiava tutta la scena.',
    'la giornata trovava un nuovo confine.', 'bastava poco per cambiare direzione.',
    'il silenzio lasciava spazio al respiro.', 'una piccola attesa attraversava lo spazio.',
    'il presente sembrava durare più a lungo.', 'gli occhi imparavano a restare fermi.',
    'qualcosa rimaneva appena fuori dalla vista.', 'il viaggio continuava dentro un pensiero.',
    'una sorpresa nasceva ai margini dello sguardo.', 'tutto sembrava vicino per la prima volta.',
    'la memoria aggiungeva un colore inatteso.', 'quel luogo chiedeva una pausa gentile.',
    'il tempo lasciava passare una promessa.', 'il giorno trovava una voce più bassa.',
    'la luce dava profondità alle cose.', 'un istante valeva quanto una giornata.',
    'lo spazio sembrava aprirsi senza rumore.', 'una calma insolita accompagnava lo sguardo.',
]


def candidates():
    for category, openings in sorted(OPENINGS.items()):
        for i, opening in enumerate(openings):
            for j, ending in enumerate(CONTINUATIONS):
                yield {'category': category, 'opening_index': i, 'ending_index': j,
                       'split': 'heldout' if (i + 3 * j) % 7 == 0 else 'train',
                       'text': opening + ', ' + ending}
