import json


def reduce_h_samples(file_in, file_out, keep_h_samples):
    with open(file_in) as file:
        source_data = file.readlines()

    source_h_samples = json.loads(source_data[0])['h_samples']
    start_index = None
    for i in range(len(source_h_samples)):
        if source_h_samples[i] == keep_h_samples[0]:
            start_index = i
            break

    with open(file_out, 'w') as file:
        for dataset in source_data:
            dict = json.loads(dataset)

            reduced_lanes = []
            for lane in dict['lanes']:
                reduced_lanes.append(lane[start_index:start_index + len(keep_h_samples)])

            file.write(json.dumps({
                'lanes': reduced_lanes,
                'h_samples': keep_h_samples,
                'raw_file': dict['raw_file']
            }) + '\n')


if __name__ == '__main__':
    reduce_h_samples('/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03/train_labels.json', '/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03/train_labels_short.json', [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
    reduce_h_samples('/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03/test.json', '/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03/test_short.json', [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
