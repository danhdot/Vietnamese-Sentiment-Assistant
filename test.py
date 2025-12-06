import json
import argparse
from transformers import pipeline

# Model duy nhất sử dụng
MODEL_NAME = 'wonrax/phobert-base-vietnamese-sentiment'

def load_tests_case(path='test_cases.json'):
    """Tải các test case từ file JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model():
    """Tải classifier PhoBERT cho tiếng Việt"""
    print(f'Sử dụng model: {MODEL_NAME}')
    model = pipeline('sentiment-analysis', model=MODEL_NAME)
    return model

def normalize_label(label):
    """Chuẩn hóa nhãn về định dạng thống nhất"""
    if label is None:
        return 'NEUTRAL'
    label_upper = label.upper()
    if 'POS' in label_upper:
        return 'POSITIVE'
    if 'NEG' in label_upper:
        return 'NEGATIVE'
    if 'NEU' in label_upper:
        return 'NEUTRAL'
    return label

def run_tests(tests_path='test_cases.json'):
    """Chạy các test case và đánh giá kết quả"""
    tests = load_tests_case(tests_path)
    model = get_model()
    total = len(tests)
    correct = 0
    results = []
    
    for index, case in enumerate(tests, start=1):
        text = case.get('text')
        expected = case.get('sentiment')
        try:
            result = model(text)
            if isinstance(result, list) and len(result) > 0:
                label = result[0].get('label')
                score = result[0].get('score', 0.0)
            else:
                label = None
                score = 0.0
        except Exception as e:
            label = None
            score = 0.0
            print(f'Lỗi ở case {index}: {e}')

        predicted = normalize_label(label)
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        results.append({'text': text, 'expected': expected, 'predicted': predicted, 'score': float(score)})
        print(f'{index:02d}. "{text}" -> kỳ vọng: {expected} ; dự đoán: {predicted} (điểm={score:.3f})')

    accuracy = correct / total * 100 if total > 0 else 0.0
    print('\nKết quả:')
    print(f'Đúng: {correct}/{total}  Độ chính xác: {accuracy:.1f}%')
    return {'total': total, 'correct': correct, 'accuracy': accuracy, 'results': results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', type=str, default='test_cases.json', help='Đường dẫn đến file JSON chứa test cases')
    args = parser.parse_args()
    run_tests(tests_path=args.tests)