import sys
import glob
import numpy as np
from music21 import *
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.callbacks import *

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def processMidi():
    sourceDir = "./songs/"
    chords = []

    songList = glob.glob(sourceDir+"*.mid")

    totalSongs = len(songList)
    songIndex = 0

    for song in songList:
        progress(songIndex, totalSongs, status=" Parsing songs")
        songIndex += 1

        midi = converter.parse(song).chordify()

        if (str(midi.analyze('key')).lower() == 'g major'):

            chordsToParse = None
            
            try:
                songList = instrument.partitionByInstrument(midi)
                chordsToParse = songList.parts[0].recurse()
            except:
                chordsToParse = midi.flat.notes

            for element in chordsToParse:
                if isinstance(element, note.Note):
                    chords.append(element.pitch)
                elif isinstance(element, chord.Chord):
                    chords.append('.'.join(str(n) for n in element.normalOrder))
    
    return chords

def prepareData(selectedChords, chordNormalize):
    sequenceLength = 100

    chordNames = sorted(set(chord for chord in selectedChords))
    chordToInt = dict((chord, number) for number, chord in enumerate(chordNames))

    networkInput = []
    networkOutput = []

    for i in range(0, len(selectedChords) - sequenceLength, 1):
        sequenceIn = selectedChords[i:i + sequenceLength]
        sequenceOut = selectedChords[i + sequenceLength]
        networkInput.append([chordToInt[note] for note in sequenceIn])
        networkOutput.append(chordToInt[sequenceOut])

    #reshape input for lstm layers
    nPatterns = len(networkInput)
    networkInput = np.reshape(networkInput, (nPatterns, sequenceLength, 1))

    #normalize input
    networkInput = networkInput / float(chordNormalize)

    networkOutput = np_utils.to_categorical(networkOutput)
    
    print('Data processed')

    return (networkInput, networkOutput)

def createNetwork(chordsInput, chordNormalize):
    model = Sequential()
    model.add(CuDNNLSTM(512, input_shape=(chordsInput.shape[1], chordsInput.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(chordNormalize))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Network generated')
    return model
    
def fitModel(model, chordsInput, chordsOutput):
    checkpoint = ModelCheckpoint(
    "weights-improvement.hdf5", monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
    )

    callbacksList = [checkpoint]
    epochNum = 2
    model.summary()
    model.fit(chordsInput, chordsOutput, callbacks=callbacksList, epochs=epochNum, batch_size=64)
    model.save('LSTMmodel.h5')
    return model

def generateChords(model, selectedChords, chordsInput, chordsNormalize):
    chordNames = sorted(set(chord for chord in selectedChords))
    intToChord = dict((number, chord) for number, chord in enumerate(chordNames))

    startingChord = np.random.randint(0, len(chordsInput)-1)
    sequence = chordsInput[startingChord]

    generatedOutput = []

    for i in range(500):
        modelInput = np.reshape(sequence, (1,len(sequence), 1))
        modelInput = modelInput / float(chordsNormalize)

        prediction = model.predict(modelInput, verbose=0)

        index = np.argmax(prediction)
        result = intToChord[index]
        generatedOutput.append(result)

        sequence = np.append(sequence, index)
        sequence = sequence[1:len(sequence)]
        
        progress(i, 500, status=' Generating Chords')

    return generatedOutput

def writeToFile(generatedChords, filename):
    offset = 0
    outputChords = []

    print()
    print('Writing midi to file')

    for pattern in generatedChords:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notesInChord = pattern.split('.')
            notes = []
            for current_note in notesInChord:
                newNote = note.Note(int(current_note))
                newNote.storedInstrument = instrument.Harpsichord()
                notes.append(newNote)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            outputChords.append(new_chord)
        # pattern is a note
        else:
            newNote = note.Note(pattern)
            newNote.offset = offset
            newNote.storedInstrument = instrument.Harpsichord()
            outputChords.append(newNote)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midiStream = stream.Stream(outputChords)
    midiStream.write('midi', fp = '{}.mid'.format(filename))

if __name__ == '__main__':
    selectedChords = processMidi()

    chordNormalize = len(set(selectedChords))

    chordsInput, chordsOutput = prepareData(selectedChords, chordNormalize)

    model = createNetwork(chordsInput, chordNormalize)

    model = fitModel(model, chordsInput, chordsOutput)

    generatedChords = generateChords(model, selectedChords, chordsInput, len(chordsInput))

    writeToFile(generatedChords, "testLSTM")
    