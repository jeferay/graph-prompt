import time

disabled_tags = set(['make batch', 'candidates_retrieve_separate', 'get_batch_inputs_for_stage_1', 'bert_candidate_generator', 'optimizer', 'get emb'])

class TimeIt:
    start = 0
    end = -1
    def __init__(self, tag='NA'):
        self.tag = tag

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, tb):
        self.end = time.time()
        if self.tag not in disabled_tags:
            print('tag={} time={:.2e}ms'.format(self.tag, (self.end-self.start)*1000))