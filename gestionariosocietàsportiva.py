import tkinter as tk
from tkinter import ttk, messagebox,PhotoImage
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime,timedelta

# Funzione per trovare un file
def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Funzione per generare il grafico maschi e femmine
def numero_atleti_per_genere_negli_anni():
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    count_per_year_gender = combined_df.groupby(['Anno', 'Sesso']).size().unstack(fill_value=0)
    count_per_year_gender.plot(kind='bar', color=['pink', 'blue'])
    plt.title('Numero di atleti per genere negli anni')
    plt.xlabel('Anno')
    plt.ylabel('Numero di atleti')
    plt.legend(title='Genere', labels=['Femmine', 'Maschi'])
    # Ottieni il manager della figura e imposta il titolo della finestra
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Numero di atleti per genere')
    plt.show()


def eta_media_per_genere():
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    mean_age_per_year_gender = combined_df.groupby(['Anno', 'Sesso'])['Eta'].mean().unstack()
    mean_age_per_year_gender.plot(kind='bar', color=['pink', 'blue'])
    plt.title('Età media per genere negli anni')
    plt.xlabel('Anno')
    plt.ylabel('Età Media')
    plt.legend(title='Genere', labels=['Femmine', 'Maschi'])
    # Ottieni il manager della figura e imposta il titolo della finestra
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Età media per genere')
    plt.show()

def calcola_punteggio_gara(valore, posizione_piu_bassa):
    if pd.isna(valore):
        return 0  # Se il valore è NaN, consideriamo il punteggio come 0
    elif valore == 1:
        return posizione_piu_bassa
    elif valore == posizione_piu_bassa:
        return 1
    else:
        return posizione_piu_bassa - valore

def risultati_per_società():
    search_path = 'csv/'
    data_list = [
        ("risultati2022.csv", "2022"),
        ("risultati2023.csv", "2023"),
        ("risultati2024.csv", "2024")
    ]

    combined_dfs = []  # Lista per memorizzare i DataFrame di tutti i file

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            combined_dfs.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not combined_dfs:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(combined_dfs)

    gare = ['Posizione Gara 1', 'Posizione Gara 2', 'Posizione Gara 3', 'Posizione Gara 4', 'Posizione Gara 5']
    punteggi_per_anno = {}

    for anno, group in combined_df.groupby('Anno'):
        punteggi_per_gara = {}
        for gara in gare:
            posizione_piu_bassa = group[gara].max()
            punteggio_gara = 0
            for valore in group[gara]:
                punteggio_gara += calcola_punteggio_gara(valore, posizione_piu_bassa)
            punteggi_per_gara[gara] = punteggio_gara
        punteggi_per_anno[anno] = punteggi_per_gara

    # Creare il grafico
    def grafico_variazione_punteggio(punteggi_per_anno):
        plt.figure(figsize=(10, 6))
        for anno, punteggi in punteggi_per_anno.items():
            plt.plot(list(punteggi.keys()), list(punteggi.values()), marker='o', linestyle='-', label=f'Anno {anno}')
        plt.title('Risultati delle gare della società negli anni')
        plt.xlabel('Gara')
        plt.ylabel('Punteggio totale')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        # Ottieni il manager della figura e imposta il titolo della finestra
        fig = plt.gcf()
        fig.canvas.manager.set_window_title('Risultati delle gare della società negli anni')
        plt.show()

    grafico_variazione_punteggio(punteggi_per_anno)

def numero_partecipanti_nelle_gare():
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("risultati2022.csv", "2022"),
        ("risultati2023.csv", "2023"),
        ("risultati2024.csv", "2024")
    ]
    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    gare = ['Posizione Gara 1', 'Posizione Gara 2', 'Posizione Gara 3', 'Posizione Gara 4', 'Posizione Gara 5']
    partecipanti_per_anno = {}

    for anno, group in combined_df.groupby('Anno'):
        numero_partecipanti_nelle_gare_anno = 0
        for gara in gare:
            partecipanti_gara = group[gara].notna().sum()
            numero_partecipanti_nelle_gare_anno += partecipanti_gara
        partecipanti_per_anno[anno] = numero_partecipanti_nelle_gare_anno

    anni_ordinati = sorted(partecipanti_per_anno.keys())
    partecipanti_ordinati = [partecipanti_per_anno[anno] for anno in anni_ordinati]
    plt.figure(figsize=(10, 6))
    plt.plot(anni_ordinati, partecipanti_ordinati, marker='o', linestyle='-', color='b')
    plt.xlabel('Anno')
    plt.ylabel('Numero di partecipanti')
    plt.title('Numero di partecipanti alle gare')
    plt.grid(True)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Numero di partecipanti alle gare')
    plt.show()

def calcola_budget(eta):
    if eta < 25:
        return 300
    else:
        return 100

def grafico_a_barre_del_budget_societario():
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]
    for filename, year in data_list:
        file_path = find_file(filename, search_path)
        
        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            print(f"File '{filename}' non trovato.")

    if not dataframes:
        print("Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    combined_df['Budget'] = combined_df['Eta'].apply(calcola_budget)

    anni = combined_df['Anno'].unique()
    budget_per_anno = {anno: combined_df[combined_df['Anno'] == anno]['Budget'].sum() for anno in anni}

    percorso_abbigliamento = find_file('abbigliamento.csv', search_path)
    df_abbigliamento = pd.read_csv(percorso_abbigliamento)
    budget_rimanente_per_anno = {anno: 0 for anno in anni}

    for anno in anni:
        anno_df = combined_df[combined_df['Anno'] == anno]
        for _, atleta in anno_df.iterrows():
            prodotti_acquistati = df_abbigliamento.sample(n=3)
            costo_prodotti = prodotti_acquistati['COSTO'].sum()
            budget_rimanente = atleta['Budget'] - costo_prodotti
            budget_rimanente_per_anno[anno] += budget_rimanente

    for anno, budget in budget_per_anno.items():
        print(f"Il budget totale per l'anno {anno} è {budget}$")
    for anno, budget in budget_rimanente_per_anno.items():
        print(f"Il budget rimanente per l'anno {anno} è {budget}$")

    plot_spese_trend(budget_per_anno, budget_rimanente_per_anno)

def plot_spese_trend(budget_iniziale_per_anno, budget_rimanente_per_anno):
    spese_totali_per_anno = {anno: budget_iniziale_per_anno[anno] - budget_rimanente_per_anno[anno] for anno in
                             budget_iniziale_per_anno}
    anni = sorted(budget_iniziale_per_anno.keys())
    budget_totale = [budget_iniziale_per_anno[anno] for anno in anni]
    spese_totali = [spese_totali_per_anno[anno] for anno in anni]

    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = range(len(anni))
    r2 = [x + bar_width for x in r1]

    ax.bar(r1, budget_totale, color='b', width=bar_width, edgecolor='grey', label='Budget Totale')
    ax.bar(r2, spese_totali, color='r', width=bar_width, edgecolor='grey', label='Spese Totali')

    ax.set_xlabel('Anno', fontweight='bold')
    ax.set_ylabel('Importo ($)', fontweight='bold')
    ax.set_title('Trend del budget e delle spese totali negli anni')
    ax.set_xticks([r + bar_width / 2 for r in range(len(anni))])
    ax.set_xticklabels(anni)
    ax.legend()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Trend del budget e delle spese totali negli anni')
    plt.show()

def plot_percentuale_spesa(ax, anno, percentuale_spesa, budget_rimanente):
    spese = [percentuale_spesa, budget_rimanente]
    labels = ['Speso', 'Rimanente']
    ax.pie(spese, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    ax.set_title(f'Anno {anno}')
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Grafico a torta del budget societario')
    ax.axis('equal')

def grafico_a_torta_del_budget_societario():
    search_path = 'csv/'
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]
    dataframes = []

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            print(f"File trovato: {file_path}")
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            print(f"File '{filename}' non trovato.")

    if not dataframes:
        print("Nessun dato trovato. Impossibile concatenare i DataFrame.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['Budget'] = combined_df['Eta'].apply(calcola_budget)

    anni = combined_df['Anno'].unique()
    budget_per_anno = {}
    budget_rimanente_per_anno = {anno: 0 for anno in anni}

    percorso_abbigliamento = find_file('abbigliamento.csv', search_path)
    df_abbigliamento = pd.read_csv(percorso_abbigliamento)

    for anno in anni:
        anno_df = combined_df[combined_df['Anno'] == anno]
        budget = anno_df['Budget'].sum()
        budget_per_anno[anno] = budget

        for _, atleta in anno_df.iterrows():
            prodotti_acquistati = df_abbigliamento.sample(n=3)
            costo_prodotti = prodotti_acquistati['COSTO'].sum()
            budget_rimanente = atleta['Budget'] - costo_prodotti
            budget_rimanente_per_anno[anno] += budget_rimanente
            print(
                f"Anno: {anno}, Atleta: {atleta['Nome']} {atleta['Cognome']}, Budget iniziale: {atleta['Budget']}, Costo prodotti: {costo_prodotti}, Budget rimanente: {budget_rimanente}")

    for anno, budget in budget_per_anno.items():
        print(f"Il budget totale per l'anno {anno} è {budget}$")
    for anno, budget in budget_rimanente_per_anno.items():
        print(f"Il budget rimanente per l'anno {anno} è {budget}$")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for idx, anno in enumerate(budget_per_anno):
        percentuale_spesa = budget_per_anno[anno] - budget_rimanente_per_anno[anno]
        plot_percentuale_spesa(axs[idx], anno, percentuale_spesa, budget_rimanente_per_anno[anno])

    plt.tight_layout()
    plt.show()


def risultati_per_atleta(selected_atleta):
    # Directory di base da cui iniziare la ricerca
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("risultati2022.csv", "2022"),
        ("risultati2023.csv", "2023"),
        ("risultati2024.csv", "2024")
    ]

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        print("Nessun dato trovato. Impossibile concatenare i DataFrame.")
        return

    # Combina tutti i DataFrame in uno solo
    df_combinato = pd.concat(dataframes, ignore_index=True)
    # print(df_combinato)

    # Filtra il DataFrame per l'atleta specifico
    atleta_df = df_combinato[df_combinato['Cognome'].str.upper() == selected_atleta.upper()]
    
    if atleta_df.empty:
        print(f"Nessun dato trovato per l'atleta '{selected_atleta}'.")
        return

    # Creazione della heatmap
    plt.figure(figsize=(10, 6))
    heatmap_data = atleta_df.drop(columns=['Nome', 'Cognome', 'Sesso', 'Eta']).set_index('Anno').fillna(-999)
    # Definisci la tua mappa di colori personalizzata
    custom_colors = ['yellow', '#FFA500', '#FF4500', 'red',
                     'darkred']  # Esempio di colori: giallo, arancione, rosso scuro
    custom_cmap = mcolors.ListedColormap(custom_colors)
    plt.imshow(heatmap_data, cmap=custom_cmap, interpolation='nearest', aspect='auto', vmin=1)
    plt.colorbar(label='Posizione', extend='min', extendrect=True, ticks=[])
    plt.xlabel('Gara')
    plt.ylabel('Anni')
    plt.title(f'Heatmap dei risultati per {selected_atleta}')
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(f'Heatmap dei risultati per {selected_atleta}')
    plt.show()

def plot_budget_rimanente_per_atleta(ax, atleta, budget_rimanente, budget_iniziale, anno):
    spese = [budget_iniziale - budget_rimanente, budget_rimanente]
    labels = ['Speso', 'Rimanente']
    ax.pie(spese, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    ax.set_title(f'{atleta} nel {anno}')
    ax.axis('equal')  # Assicurarsi che il grafico sia un cerchio

def grafico_budget_rimanente_per_atleta(selected_atleta):
    # Directory di base da cui iniziare la ricerca
    search_path = 'csv/'
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]
    dataframes = []
    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        print("Nessun dato trovato. Impossibile concatenare i DataFrame.")
        return

    # Combina tutti i DataFrame in uno solo
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['Budget'] = combined_df['Eta'].apply(calcola_budget)

    # Ottieni tutti gli anni presenti nel DataFrame
    anni = combined_df['Anno'].unique()
    # Calcola il budget totale per ogni anno e per ogni atleta
    budget_per_anno = {}
    budget_per_atleta_per_anno = {}
    budget_rimanente_per_atleta_per_anno = {anno: {} for anno in anni}

    percorso_abbigliamento = find_file('abbigliamento.csv', search_path)
    df_abbigliamento = pd.read_csv(percorso_abbigliamento)

    for anno in anni:
        anno_df = combined_df[combined_df['Anno'] == anno]
        budget = anno_df['Budget'].sum()
        budget_per_anno[anno] = budget
        budget_per_atleta = anno_df.groupby(['Cognome'])['Budget'].sum().to_dict()
        budget_per_atleta_per_anno[anno] = budget_per_atleta

        # Calcola il budget rimanente per ogni atleta e per ogni anno
        for atleta, budget_iniziale in budget_per_atleta.items():
            # Seleziona alcuni prodotti casuali per l'acquisto
            prodotti_acquistati = df_abbigliamento.sample(n=3)
            # Calcola il costo totale dei prodotti acquistati
            costo_prodotti = prodotti_acquistati['COSTO'].sum()
            # Sottrai il costo dei prodotti acquistati dal budget dell'atleta
            budget_rimanente = budget_iniziale - costo_prodotti
            # Aggiungi il budget rimanente per questo atleta al totale per l'anno
            budget_rimanente_per_atleta_per_anno[anno][atleta] = budget_rimanente
            # print(f"Anno: {anno}, Atleta: {atleta}, Budget iniziale: {budget_iniziale}, Costo prodotti: {costo_prodotti}, Budget rimanente: {budget_rimanente}")

    atleta_selezionato = selected_atleta

    # Verifica se l'atleta selezionato esiste nei dati
    atleta_trovato = False
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_index = 0

    for anno, budget_rimanente_per_atleta in budget_rimanente_per_atleta_per_anno.items():
        if atleta_selezionato in budget_rimanente_per_atleta:
            atleta_trovato = True
            budget_rimanente = budget_rimanente_per_atleta[atleta_selezionato]
            budget_iniziale = budget_per_atleta_per_anno[anno][
                atleta_selezionato]  # Ottieni il budget iniziale corretto
            plot_budget_rimanente_per_atleta(axs[plot_index], atleta_selezionato, budget_rimanente, budget_iniziale,
                                             anno)
            plot_index += 1

    if not atleta_trovato:
        print("atleta non trovato")
    else:
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(f'Grafico budget rimanente dell\'atleta {selected_atleta}')
        plt.show()

def carica_atleti():
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("anagrafica2022.csv", "2022"),
        ("anagrafica2023.csv", "2023"),
        ("anagrafica2024.csv", "2024")
    ]

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    nomi_atleti = combined_df['Cognome'].unique()
    return nomi_atleti.tolist()

def partecipanti_gare_future():
    search_path = 'csv/'
    filename='gare2025.csv'
    file_path= find_file(filename,search_path)
    print(file_path)
    if file_path:
        df=pd.read_csv(file_path)
        df_per_stampa=df.to_string() # per prendere tutte le stringhe
        #print(df_per_stampa)
    else:
        print('no')
    gare=['Gara1','Gara2','Gara3','Gara4','Gara5']
    numero_partecipanti=[]
    non_partecipanti=[]
    for gara in gare:
        if gara in df.columns:
            numero_partecipanti.append(df[gara].str.count('SI').sum())
            non_partecipanti.append(df[gara].str.count('NO').sum())
        else:
            print(f"Colonna {gara} non trovata nel file CSV")
    
    #print(f"Numero totale di iscritti: {numero_partecipanti+non_partecipanti}, dei quali partecipano {numero_partecipanti},mentre non partecipano {non_partecipanti}")
    # Generare il grafico a barre raggruppate
    x = np.arange(len(gare))  # la posizione delle barre
    width = 0.35  # la larghezza delle barre
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width/2,numero_partecipanti, width, label='Partecipanti', color='yellow')
    bar2 = ax.bar(x + width/2, non_partecipanti, width, label='Non Partecipanti', color='green')
    # Aggiungere etichette, titolo e legenda
    ax.set_xlabel('Gare')
    ax.set_ylabel('Numero di Persone')
    ax.set_title('Partecipanti e Non Partecipanti per le prossime gare')
    ax.set_xticks(x)
    ax.set_xticklabels(gare)
    ax.legend()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Numero di partecipanti alle prossime gare')
    plt.show()

def pagamenti():
    search_path = 'csv/'
    filename = 'pagamenti.csv'
    file_path = find_file(filename, search_path)
    if file_path:
        df = pd.read_csv(file_path)

        # Filtrare il DataFrame per ottenere solo le righe con pagamento non effettuato
        #df = df[df['Pagamento Effettuato'] == 'No']

        if not df.empty:
            tabella(df)
        else:
            messagebox.showinfo("Informazione", "Tutti hanno effettuato il pagamento")
    else:
        messagebox.showerror("Errore", "File non trovato")

def tabella(df):
    window = tk.Tk()
    window.title("Tabella pagamenti")
    frame = ttk.Frame(window, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    # Creare una tabella
    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show='headings')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    # Configurare la scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    window.mainloop()

# Funzione per eseguire la funzione selezionata
def execute_selected_function():
    selected_function = function_var.get()
    if selected_function == "Numero di atleti per genere negli anni":
        numero_atleti_per_genere_negli_anni()
    elif selected_function == "Età media per genere":
        eta_media_per_genere()
    elif selected_function == "Risultati per società":
        risultati_per_società()
    elif selected_function == "Numero di partecipanti alle gare":
        numero_partecipanti_nelle_gare() 
    elif selected_function == "Risultati atleta":
        selected_atleta = atleta_var.get()  # Ottieni il nome dell'atleta selezionato
        risultati_per_atleta(selected_atleta)  # Chiama la funzione risultati_per_atleta con il nome dell'atleta
        # Chiamiamo la funzione per mostrare la tabella dei partecipanti
        # Passiamo il DataFrame df_partecipanti creato dalla tua funzione nomi_partecipanti()
        nomi_partecipanti_df = nomi_partecipanti(selected_atleta)
        mostra_tabella_partecipanti(nomi_partecipanti_df)
    elif selected_function == "Grafico a barre del budget societario":
        grafico_a_barre_del_budget_societario()
    elif selected_function == "Grafico a torta del budget societario":
        grafico_a_torta_del_budget_societario()
    elif selected_function == "Budget atleta":
        selected_atleta = atleta_var.get()
        grafico_budget_rimanente_per_atleta(selected_atleta)
    elif selected_function == 'Numero di partecipanti alle prossime gare':
        partecipanti_gare_future()
    elif selected_function == 'Tabella pagamenti':
        pagamenti()
    elif selected_function == 'Tabella certificati medici':
        certificatomedico()
    else:
        messagebox.showerror("Errore", "Funzione non valida selezionata.")

# Funzione per determinare lo stato del certificato
def verifica_validita_certificato(data_scadenza):
    oggi = datetime.now().date()
    data_scadenza = pd.to_datetime(data_scadenza).date()
    if data_scadenza < oggi:
        return 'Invalido'
    elif data_scadenza <= oggi + timedelta(days=60):
        return 'In scadenza'
    else:
        return 'Valido'

def certificatomedico():
    search_path='csv/'
    file_path=find_file('CertificatoMedico.csv',search_path)
    if file_path is None:
        messagebox.showerror('Errore','Il file non è stato trovato')
    else:
        df=pd.read_csv(file_path)
        # Applicare la funzione al DataFrame per creare una nuova colonna 'stato_certificato'
        df['Certificato Medico'] = df['Data Scadenza'].apply(verifica_validita_certificato)
        
    def tabella(df):
        window = tk.Tk()
        window.title("Tabella dei certificati medici")
        frame = ttk.Frame(window, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Creare una tabella
        cols = list(df.columns)
        tree = ttk.Treeview(frame, columns=cols, show='headings')
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        # Configurare la scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        # Funzione per mostrare l'errore dopo 10 secondi
        def mostra_errore():
            cert_in_scadenza = df[df['Certificato Medico'] == 'In scadenza']
            if not cert_in_scadenza.empty:
                message = "I seguenti certificati sono in scadenza:\n" + cert_in_scadenza.to_string(index=False)
                messagebox.showwarning("Certificati in scadenza", message)
            window.lift()  # Porta la finestra in primo piano
            
        # Chiamare mostra_errore dopo 10 secondi (10000 millisecondi)
        window.after(3000, mostra_errore)
        window.mainloop()
        
    tabella(df)
       
def nomi_partecipanti(selected_atleta):
    search_path = 'csv/'
    dataframes = []
    data_list = [
        ("risultati2022.csv", "2022"),
        ("risultati2023.csv", "2023"),
        ("risultati2024.csv", "2024")
    ]
    partecipanti_info = []  # Inizializza la lista dei partecipanti_info qui

    for filename, year in data_list:
        file_path = find_file(filename, search_path)

        if file_path:
            df = pd.read_csv(file_path)
            df['Anno'] = year
            dataframes.append(df)
        else:
            messagebox.showerror("Errore", f"File '{filename}' non trovato.")

    if not dataframes:
        messagebox.showerror("Errore", "Nessun file valido trovato.")
        return

    combined_df = pd.concat(dataframes)
    combined_df=combined_df[combined_df['Cognome']==selected_atleta]
    gare = ['Posizione Gara 1', 'Posizione Gara 2', 'Posizione Gara 3', 'Posizione Gara 4', 'Posizione Gara 5']
    for anno, group in combined_df.groupby('Anno'):
        for gara in gare:
            # Filtra solo le righe con valori non nulli nella colonna della gara attuale
            partecipanti_gara = group[group[gara].notna()]
            # Estrai nome, cognome, gara e posizione e aggiungili alla lista partecipanti_info
            for index, row in partecipanti_gara.iterrows():
                anno = row['Anno']
                nome = row['Nome']
                cognome = row['Cognome']
                posizione = row[gara]
                partecipanti_info.append({'Anno': anno,'Nome': nome, 'Cognome': cognome, 'Gara': gara, 'Posizione': posizione})

    # Creiamo un DataFrame pandas con le informazioni dei partecipanti
    df_partecipanti = pd.DataFrame(partecipanti_info)

    # Ordiniamo il DataFrame per anno e posizione
    df_partecipanti = df_partecipanti.sort_values(by=['Anno', 'Posizione'])
    return df_partecipanti

def mostra_tabella_partecipanti(df_partecipanti):
    # Creiamo una finestra tkinter
    root = tk.Tk()
    root.title("Tabella del partecipante")
    # Creiamo una tabella tkinter con i dati del DataFrame
    tabella = ttk.Treeview(root)
    tabella["columns"] = list(df_partecipanti.columns)
    tabella["show"] = "headings"
    # Aggiungiamo le colonne alla tabella
    for col in df_partecipanti.columns:
        tabella.heading(col, text=col)
    # Aggiungiamo le righe alla tabella
    for index, row in df_partecipanti.iterrows():
        tabella.insert("", "end", values=list(row))
    # Aggiungiamo la tabella alla finestra
    tabella.pack(expand=True, fill="both")
    # Avviamo il loop principale della finestra tkinter
    root.mainloop()

def on_function_select(event):
    global window_id_atleta_label
    global window_id_atleta_menu
    selection = function_var.get()
    if selection == "Risultati atleta" or selection == "Budget atleta":
        global atleta_var
        atleta_var = tk.StringVar()
        atleta_label = tk.Label(root, text="Seleziona un atleta:", font=("Helvetica", 16))
        atleta_menu = ttk.Combobox(root, textvariable=atleta_var, state="readonly", font=("Helvetica", 16))
        nomi_atleti = carica_atleti()
        # Popolare il menu a tendina con i nomi degli atleti disponibili
        atleta_menu['values'] = nomi_atleti
        window_id_atleta_label = canvas.create_window(750, 400, window=atleta_label, anchor="center")  # Posizionamento della label nel canvas
        window_id_atleta_menu = canvas.create_window(750, 450, window=atleta_menu, anchor="center")
    else:
        canvas.delete(window_id_atleta_label)
        canvas.delete(window_id_atleta_menu)
        root.bind("<Configure>", resize_background)
        
def resize_background(event):
    global background_photo, background_image, canvas_image_id
    # Ottieni le nuove dimensioni della finestra
    new_width = event.width
    new_height = event.height
    # Ridimensiona l'immagine di sfondo
    resized_image = background_image.resize((new_width, new_height), Image)
    background_photo = ImageTk.PhotoImage(resized_image)
    # Aggiorna l'immagine del canvas
    canvas.itemconfig(canvas_image_id, image=background_photo)
    canvas.image = background_photo  # Mantieni un riferimento all'immagine per evitare che venga garbage collected

# Creazione della finestra principale
root = tk.Tk()
root.title("Progetto Gruppo 1")
root.geometry("1920x1080")  # Imposta le dimensioni della finestra a 1920x1080
root.state('zoomed')
# Caricamento dell'icona
icon_path = "icon.png"
icon_image = Image.open(icon_path)
icon_photo = ImageTk.PhotoImage(icon_image)
root.iconphoto(True, icon_photo)
# Caricamento dello sfondo
background_image_path = "sfondo66.jpg"
background_image = Image.open(background_image_path)
background_photo = ImageTk.PhotoImage(background_image)
# Creazione del frame principale
main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill=tk.BOTH)
# Creazione del canvas per lo sfondo
canvas = tk.Canvas(main_frame, highlightthickness=0)
canvas.pack(fill="both", expand=True)
# Aggiunta dello sfondo al canvas
canvas_image_id = canvas.create_image(0, 0, anchor="nw", image=background_photo)
# Caricamento e ridimensionamento delle immagini
top6uomini_image = Image.open("top6uomini2.jpg")
top6donne_image = Image.open("top6donne2.jpg")
desired_width = 330  # Larghezza desiderata per le immagini
desired_height = 400  # Altezza desiderata per le immagini
top6uomini_image = top6uomini_image.resize((desired_width, desired_height))
top6donne_image = top6donne_image.resize((desired_width, desired_height))
# Caricamento delle immagini
top6uomini_photo = ImageTk.PhotoImage(top6uomini_image)
top6donne_photo = ImageTk.PhotoImage(top6donne_image)
# Posizionamento delle immagini sul canvas
canvas.create_image(0, 400, anchor="nw", image=top6uomini_photo)
canvas.create_image(canvas.winfo_width() - desired_width - 50, 50, anchor="nw", image=top6donne_photo)
# Posizionamento delle immagini sul canvas
canvas.create_image(0, 0, anchor="nw", image=top6donne_photo)
canvas.create_image(canvas.winfo_width() - desired_width - 50, 50, anchor="nw", image=top6donne_photo)
# Aggiornamento dello sfondo quando la finestra viene ridimensionata
canvas.bind("<Configure>", resize_background)
# Etichetta del titolo
title_label = tk.Label(main_frame, text="Gestione Società Sportiva", bg='#e35c2b', fg="black", font=("Helvetica", 23, "bold"))
title_label.place(relx=0.5, rely=0.1, anchor="center")
# Menu a tendina per selezionare la funzione
function_var = tk.StringVar()
function_menu = ttk.Combobox(main_frame, textvariable=function_var, state="readonly", font=("Helvetica", 16), width=32)
function_menu['values'] = ("Numero di atleti per genere negli anni", "Età media per genere", "Risultati per società",
                            "Numero di partecipanti alle gare", "Risultati atleta",
                            "Grafico a barre del budget societario", "Grafico a torta del budget societario",
                            "Budget atleta", "Numero di partecipanti alle prossime gare","Tabella pagamenti",
                            'Tabella certificati medici')
function_menu.set("Seleziona una funzione")
function_menu.place(relx=0.5, rely=0.25, anchor="center")
function_menu.bind("<<ComboboxSelected>>", on_function_select)
# Pulsante per eseguire la funzione selezionata
execute_btn = tk.Button(root, text="Esegui Funzione", command=execute_selected_function, font=("Helvetica", 18),
                        bg='#e35c2b', width=23, height=1)
execute_btn.place(relx=0.5, rely=0.35, anchor="center")
# Avvio del loop principale di Tkinter
root.mainloop()
