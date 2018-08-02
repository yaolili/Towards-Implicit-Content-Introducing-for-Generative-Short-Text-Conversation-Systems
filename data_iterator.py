import numpy

import pickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, topic,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.topic = fopen(topic, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.topic_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.topic.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        topic = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer) and len(self.topic_buffer) == len(self.source_buffer) , 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                kw = self.topic.readline()
                if kw == "":
                    break
                
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
                self.topic_buffer.append(kw.strip().split())

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]
            _kwbuf = [self.topic_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf
            self.topic_buffer = _kwbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.topic_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                # read from topic file and map to word index
                kw = self.topic_buffer.pop()
                kw = [self.target_dict[w] if w in self.target_dict else 1
                      for w in kw]
                if self.n_words_target > 0:
                    kw = [w if w < self.n_words_target else 1 for w in kw]
                    
                if len(ss) > self.maxlen and len(tt) > self.maxlen and len(kw) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
                topic.append(kw)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(topic) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, topic
