import torch
from torch.utils.data import Dataset
import torchaudio
import helpers

class AudioMnistPowSpec(Dataset):

    def __init__(self,audiodir_path,sr,sig_len,N_speakers,snr):
        self.audio_path=audiodir_path
        self.N_speakers=N_speakers
        self.sr=sr
        self.snr=snr
        self.sig_len=sig_len
        self.Name, self.Label = self.create_datapoints()

    def create_datapoints(self):
        Name=[]
        Label=[]
        for spk_idx in range(1,self.N_speakers):
            for digit_idx in range (0,10):
                for utt_idx in range (0,50):
                    audiofilename = self.audio_path + "/%02d/%d_%02d_%d.wav" % (spk_idx, digit_idx, spk_idx, utt_idx)
                    label = digit_idx
                    # append data set
                    Name.append(audiofilename)
                    Label.append(label)
        return Name, Label

    def __len__(self):
        return len(self.Name)

    def __getitem__(self,index):
        # load signal
        _,S,_=helpers.wav2powspec(filename=self.Name[index], n_fft=1024, hop_length=512, win_length = None, sample_rate = self.sr, pad_dur=1)
        data_point=S
        label=self.Label[index]
        return data_point, label

#if __name__ == "__main__":
