import argparse
import sys
sys.path.insert(0, 'TW_FR_EN_corpus/scripts')
from googleAPI import GooglePivot
from preprocessing import Preprocessing
import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Perform Pivot Translation From Source Language to Target Language Via A Third Language With Google Translate API')
    parser.add_argument("file",
                        help="Provide The Source Language To Be Translated in a TXT File", type=str)
    parser.add_argument("src_key",
                        help="Provide The Google Translate API Key For The Source Language", type=str)
    parser.add_argument("dest_key",
                        help="Provide The Google Translate API Key For The Destination Language", type=str)
    parser.add_argument("--output_name",
                        help="Pass Path And Name Of The Final Output File", default="translate.txt", type=str)

    parser.add_argument("--to_console",
                        help="Specify As True If You Want To Print The Output TO The Console.", default=False, type=bool)

    args = parser.parse_args()

    translator = GooglePivot()
    preprocessor = Preprocessing()

    if args.to_console == True:
        with open(args.file) as txt_file:
            for line in txt_file:
                src = line.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                print(f'Source: {src.strip().capitalize()}')
                translated_text = translator.evaluate(
                    line, args.src_key, args.dest_key)
                translated_text = translated_text.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")
                print(f'Target: {translated_text.strip().capitalize()}')
                print()
    else:
        lines = []
        with open(args.file) as txt_file:
            for line in txt_file:
                translated_text = translator.evaluate(
                    line.strip(), args.src_key, args.dest_key)
                translated_text = translated_text.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")\
                    .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")\
                    .replace(" ) ", ") ").replace(" , ", ", ")                
                lines.append(translated_text.strip().capitalize())

        preprocessor.writeTotxt(args.output_name, lines)


    