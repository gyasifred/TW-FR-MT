from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
import argparse
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split corpus into Training, validation and Test sets')
    parser.add_argument(
        'filepath_1', metavar='filepath_1', help="Enter the file path of Twi document", type=str)
    parser.add_argument(
        'filepath_2', metavar='filepath_2', help="Enter the file path of English document", type=str)
    parser.add_argument(
        'filepath_3', metavar='filepath_3', help="Enter the file path of French document", type=str)
    parser.add_argument(
        '--id_tw', default="tw", help="Provide the key for Twi. Example tw for Twi")
    parser.add_argument('--id_en', default="en",
                        help="Provide the key for English.")
    parser.add_argument(
        '--id_fr', default="fr", help="Provide the key for French")
    parser.add_argument('--val_set', default=False, type=bool,
                        help='Pass True if validation set is required. Default = False')
    parser.add_argument('--train_ouput_path', default='.',
                        help='Output path for training  set')
    parser.add_argument(
        '--val_ouput_path', default='.', help='Output path for validation  set')
    parser.add_argument('--test_ouput_path', default='.',
                        help='Output path for test  set')

    args = parser.parse_args()

    # Create an instance of preprocessing class
    preprocessor = Preprocessing()

    # Read raw parallel dataset
    raw_data_twi, raw_data_en, raw_data_fr = preprocessor.read_parallel_dataset(
        filepath_1=args.filepath_1,
        filepath_2=args.filepath_2,
        filepath_3=args.filepath_3
    )

    # Normalize the raw data
    raw_data_en = [preprocessor.normalize_FrEn(data) for data in raw_data_en]
    raw_data_twi = [preprocessor.normalize_twi(data) for data in raw_data_twi]
    raw_data_fr = [preprocessor.normalize_FrEn(data) for data in raw_data_fr]

    # split the dataset into training and test sets
    # Keep 20% of the data as test set and validation
    if (args.val_set == True):

        train_twi, test_twi, train_en, test_en, train_fr, test_fr = train_test_split(
            raw_data_twi, raw_data_en, raw_data_fr, test_size=0.2, random_state=42)

        # split test into testans validation
        test_twi, val_twi, test_en, val_en, test_fr, val_fr = train_test_split(
            test_twi, test_en, test_fr, test_size=0.5, random_state=42)

        # write the preprocess traning and test dataset to a file
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_tw}.txt', train_twi)
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_en}.txt', train_en)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_tw}.txt', test_twi)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_en}.txt', test_en)
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_fr}.txt', train_fr)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_fr}.txt', test_fr)
        preprocessor.writeTotxt(
            f'{args.val_ouput_path}/val_{args.id_fr}.txt', val_fr)
        preprocessor.writeTotxt(
            f'{args.val_ouput_path}/val_{args.id_tw}.txt', val_twi)
        preprocessor.writeTotxt(
            f'{args.val_ouput_path}/val_{args.id_en}.txt', val_en)
    else:
        train_twi, test_twi, train_en, test_en, train_fr, test_fr = train_test_split(
            raw_data_twi, raw_data_en, raw_data_fr, test_size=0.1, random_state=42)

        # write the preprocess traning and test dataset to a file
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_tw}.txt', train_twi)
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_en}.txt', train_en)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_tw}.txt', test_twi)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_en}.txt', test_en)
        preprocessor.writeTotxt(
            f'{args.train_ouput_path}/train_{args.id_fr}.txt', train_fr)
        preprocessor.writeTotxt(
            f'{args.test_ouput_path}/test_{args.id_fr}.txt', test_fr)

    print("SUCCESS")