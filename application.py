import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as ndimage
import pandas as pd
import os
import tempfile


# ADDSONGS

DATABASE_SONGS = [
    "A Day In The Life.mp3" , 
    "A Hard Day_s Night.mp3" , 
    "Across The Universe.mp3" , 
    "Back In The U.S.S.R..mp3" , 
    "Blackbird.mp3" , 
    "Bohemian Rhapsody.mp3" , 
    "Can_t Buy Me Love.mp3" , 
    "Crazy Little Thing Called Love.mp3" , 
    "Day Tripper.mp3" , 
    "Don_t Stop Me Now.mp3" , 
    "Drive My Car.mp3" , 
    "Eight Days A Week.mp3" , 
    "Eleanor Rigby.mp3" , 
    "Get Back.mp3" , 
    "Hello, Goodbye.mp3" , 
    "Help!.mp3" , 
    "Helter Skelter.mp3" , 
    "Hey Jude.mp3" , 
    "I Am The Walrus.mp3" , 
    "I Saw Her Standing There.mp3" , 
    "I Want It All.mp3" , 
    "I Want To Hold Your Hand.mp3" , 
    "In My Life.mp3" , 
    "I_ll Follow The Sun.mp3" , 
    "I_ve Got A Feeling.mp3" , 
    "Killer Queen.mp3" , 
    "Let It Be.mp3" , 
    "Love Me Do.mp3" , 
    "Lucy In The Sky With Diamonds.mp3" , 
    "Never Gonna Give You Up.mp3" , 
    "Norwegian Wood (This Bird Has Flown).mp3" , 
    "Penny Lane.mp3" , 
    "Radio Ga Ga.mp3" , 
    "Revolution.mp3" , 
    "Sgt. Pepper_s Lonely Hearts Club Band.mp3" , 
    "She Said She Said.mp3" , 
    "Somebody To Love.mp3" , 
    "Something.mp3" , 
    "Taxman.mp3" , 
    "The Long And Winding Road.mp3" , 
    "Two Of Us.mp3" , 
    "Under Pressure.mp3" , 
    "We Are The Champions.mp3" , 
    "We Can Work It Out.mp3" , 
    "We Will Rock You.mp3" , 
    "While My Guitar Gently Weeps.mp3" , 
    "With A Little Help From My Friends.mp3" , 
    "Within You Without You.mp3" , 
    "Yesterday.mp3" , 
    "You Really Got A Hold On Me.mp3"
]


# ALGOFUNCTS


def extract_peaks(audio, fs, sq=50, segLen=4096):
    """Calculates the spectrogram and finds constellation peaks."""
    frequencies, time, power = signal.spectrogram(audio, fs=fs, nperseg=segLen)
    powerdB = 10 * np.log10(np.maximum(power, 1e-10))

    localMax = ndimage.maximum_filter(powerdB, size=(sq, sq))
    peakMask = (powerdB == localMax)
    threshold = np.max(powerdB) - 50.0
    amplitudeFilterMask = (powerdB > threshold)
    finalMask = amplitudeFilterMask & peakMask

    freqIndex, timeIndex = np.where(finalMask)
    peakTimes = time[timeIndex]
    peakFrequencies = frequencies[freqIndex]

    sortIndices = np.argsort(peakTimes)
    peaks = list(zip(peakTimes[sortIndices], peakFrequencies[sortIndices]))

    return peaks, time, frequencies, powerdB

def generate_hashes(peaks, name, fanValue=15):
    """Generates hash pairs from a list of peaks."""
    hashes = []
    for i in range(len(peaks)):
        anchorTime, anchorFreq = peaks[i]
        for j in range(1, fanValue + 1):
            if i + j < len(peaks):
                targetTime, targetFreq = peaks[i + j]
                deltaTime = targetTime - anchorTime
                hashKey = (int(anchorFreq), int(targetFreq), round(deltaTime, 3))
                hashValue = (name, round(anchorTime, 3))
                hashes.append((hashKey, hashValue))
    return hashes

def identify_clip(clipHashes, database):
    """Matches clip hashes against the database and tallies offsets."""
    matchCounts = {}
    for qHashKey, qHashValue in clipHashes:
        _, qTime = qHashValue
        if qHashKey in database:
            for DBSongName, DBtime in database[qHashKey]:
                offset = round(DBtime - qTime, 2)
                if DBSongName not in matchCounts:
                    matchCounts[DBSongName] = {}
                if offset not in matchCounts[DBSongName]:
                    matchCounts[DBSongName][offset] = 0
                matchCounts[DBSongName][offset] += 1
                
    bestSong = None
    maxMatches = 0
    bestOffset = 0
    for song, offsets in matchCounts.items():
        for offset, count in offsets.items():
            if count > maxMatches:
                maxMatches = count
                bestSong = song
                bestOffset = offset

    return bestSong, maxMatches, bestOffset, matchCounts


# VISUALIZATIONFUNCS


def plot_spectro_and_constellation(time, frequencies, powerdB, peaks, name):
    """Creates the superimposed spectrogram and constellation plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    mesh = ax.pcolormesh(time, frequencies, powerdB, shading='gouraud', cmap='magma')
    
    peakTimes = [p[0] for p in peaks]
    peakFreqs = [p[1] for p in peaks]
    
    ax.scatter(peakTimes, peakFreqs, color='yellow', s=15, marker='x', label='Peaks')
    ax.set_title(f"Spectrogram & Constellation: {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 8000)
    fig.colorbar(mesh, ax=ax, label='Power (dB)')
    ax.legend(loc='upper right')
    return fig

def plot_offset_histogram(matchCounts, bestSong):
    """Plots the histogram of time offsets to prove the match."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if bestSong and bestSong in matchCounts:
        offsets = list(matchCounts[bestSong].keys())
        counts = list(matchCounts[bestSong].values())
        
        ax.bar(offsets, counts, width=0.2, color='#2ca02c', edgecolor='black')
        ax.set_title(f"Offset Histogram for '{bestSong}'")
        ax.set_xlabel("Time Offset (Seconds)")
        ax.set_ylabel("Number of Matching Hashes")
        ax.grid(axis='y', linestyle=':', alpha=0.7)
    return fig


#STREAMLITCODE


#PageLayout
st.set_page_config(page_title="Q3B Submission", layout="wide")
st.title("Audio Fingerprinting Application")
st.subheader("ANIMESH SRIVASTAVA [250134]")

#CacheImplementation Done.
@st.cache_resource(show_spinner=False)
def init_database():
    database = {}
    valid_songs = []
    
    with st.spinner("Building the song database from configuration..."):
        for song_file in DATABASE_SONGS:
            if os.path.exists(song_file):
                # Standardize to 22050Hz for speed and memory efficiency
                audio, sr = librosa.load(song_file, sr=22050)
                peaks, _, _, _ = extract_peaks(audio, sr)
                song_hashes = generate_hashes(peaks, song_file)
                
                for h_key, h_val in song_hashes:
                    if h_key not in database:
                        database[h_key] = []
                    database[h_key].append(h_val)
                valid_songs.append(song_file)
            else:
                st.warning(f"File not found: {song_file}. Please ensure it is in the same folder.")
                
    return database, valid_songs

db, loaded_songs = init_database()

if not loaded_songs:
    st.error("No database songs found! Please add valid files to `DATABASE_SONGS` in the code.")
    st.stop()

st.success(f"Database loaded successfully with {len(loaded_songs)} songs!")


def load_uploaded_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    audio, sr = librosa.load(tmp_path, sr=22050)
    os.remove(tmp_path)
    return audio, sr


# UI (NeedsMoreWork)


tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

# single done.
with tab1:
    st.markdown("Upload a single noisy clip to identify the song and view the intermediate steps.")
    query_file = st.file_uploader("Upload Query Clip", type=['mp3', 'wav'], key="single_upload")
    
    if query_file and st.button("Identify Clip", key="single_btn"):
        with st.spinner("Analyzing audio..."):
            #Load Audio
            q_audio, q_sr = load_uploaded_audio(query_file)
            
            #Extract Data
            q_peaks, q_time, q_freqs, q_power = extract_peaks(q_audio, q_sr)
            q_hashes = generate_hashes(q_peaks, query_file.name)
            
            #Identify
            best_song, matches, offset, match_counts = identify_clip(q_hashes, db)
            
            if best_song:
                #(filename without extension) done.
                prediction = os.path.splitext(best_song)[0]
                
                st.success("Analysis Complete!")
                st.markdown(f"### Matched Song: **{prediction}**")
                st.markdown(f"*Aligned at **{offset} seconds** with **{matches}** perfectly matching hash pairs.*")
                
                st.markdown("---")
                st.subheader("Intermediate Steps")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**1. Spectrogram & Constellation**")
                    fig_spec = plot_spectro_and_constellation(q_time, q_freqs, q_power, q_peaks, "Query Clip")
                    st.pyplot(fig_spec)
                
                with col2:
                    st.markdown("**2. Offset Histogram**")
                    st.markdown("A massive spike indicates the exact alignment offset.")
                    fig_hist = plot_offset_histogram(match_counts, best_song)
                    st.pyplot(fig_hist)
            else:
                st.error("No match found in the database. Try another clip!")

# bat done.
with tab2:
    st.markdown("Upload multiple query clips to generate a standardized `results.csv` for automatic evaluation.")
    batch_files = st.file_uploader("Upload Query Clips (Multiple allowed)", type=['mp3', 'wav'], accept_multiple_files=True, key="batch_upload")
    
    if batch_files and st.button("Run Batch Evaluation", key="batch_btn"):
        results = []
        progress_bar = st.progress(0)
        
        with st.spinner("Processing batch..."):
            for i, file in enumerate(batch_files):


                q_audio, q_sr = load_uploaded_audio(file)
                q_peaks, _, _, _ = extract_peaks(q_audio, q_sr)
                q_hashes = generate_hashes(q_peaks, file.name)
                
                best_song, _, _, _ = identify_clip(q_hashes, db)
                
                # Format as requested: filename without extension
                prediction = os.path.splitext(best_song)[0] if best_song else "No Match"
                
                results.append({
                    "filename": file.name,
                    "prediction": prediction
                })
                
                progress_bar.progress((i + 1) / len(batch_files))
                
        # GiveCSV
        results_df = pd.DataFrame(results)
        st.success("Batch processing complete!")
        st.dataframe(results_df, use_container_width=True)
        
        # DownloadCSV
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download results.csv",
            data=csv_data,
            file_name="results.csv",
            mime="text/csv"
        )