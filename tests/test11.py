from tensorflow.python.summary.summary_iterator import summary_iterator

for event in summary_iterator('/home/loren/PycharmProjects/blogpost/tests/tensorboard/A2C_1/events.out.tfevents.1674580385.loren-MS-7C91.23819.0'):
    for value in event.summary.value:
        print(value.tag)
        if value.HasField('simple_value'):
            print(value.simple_value)