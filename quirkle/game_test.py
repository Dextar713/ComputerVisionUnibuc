from quirkle.feature_extractor import encode_game, write_config, write_score, set_input_output_paths
from quirkle.position_locator import decode_config
import os

mode = 'fake_test'
# mode = 'train'

def run_tests(input_img_path: str, output_conf_path: str,
              tests_set_cnt: int, tests_cnt: int, eval_mode: bool = True, input_conf_path: str = '') -> None:
    num_errors = 0
    set_input_output_paths(input_img_path, output_conf_path)
    for test_set_num in range(1, tests_set_cnt + 1):
        for test_num in range(1, tests_cnt + 1):
    # for test_set_num in range(1, 5 + 1):
    #     for test_num in range(1, 20 + 1):
            start_img_name = f'{test_set_num}_00.jpg'
            prev_img_name = f'{test_set_num}_{test_num - 1:02d}.jpg'
            cur_img_name = f'{test_set_num}_{test_num:02d}.jpg'

            try:
                start_game, start_config = encode_game(prev_img_name, start_img_name)
                cur_game, cur_config = encode_game(cur_img_name, start_img_name)
                encoded_diff = cur_config - start_config
                diff_config = decode_config(encoded_diff)
                cur_score = cur_game.calculate_game_score(encoded_diff)
                #print('Current score:', cur_score)
                write_config(diff_config, test_set_num, test_num)
                write_score(cur_score, test_set_num, test_num)
            except Exception as e:
                num_errors += 1
                print(e)
                #print(decode_config(start_config)[3][14])
                #print(decode_config(cur_config)[3][14])
                print(f'Error test case {test_set_num}_{test_num:02d}')
                continue

            if not eval_mode:
                continue
            try:
                cur_dir = os.path.dirname(__file__)
                # if mode == 'train':
                #     input_file = open(f'antrenare/{test_set_num}_{test_num:02d}.txt', 'r')
                #     output_file = open(f'train_output/{test_set_num}_{test_num:02d}.txt', 'r')
                # elif mode == 'fake_test':
                #     input_file = open(f'evaluare/fake_test/ground-truth/{test_set_num}_{test_num:02d}.txt', 'r')
                #     output_file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'r')
                # else:
                #     input_file = open(f'evaluare/fake_test/ground-truth/{test_set_num}_{test_num:02d}.txt', 'r')
                #     output_file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'r')
                input_file = open(os.path.join(cur_dir, input_conf_path, f'{test_set_num}_{test_num:02d}.txt'), 'r')
                output_file = open(os.path.join(cur_dir, output_conf_path, f'{test_set_num}_{test_num:02d}.txt'), 'r')

                #input_lines = input_file.readlines()[:-1]
                input_lines = input_file.readlines()
                output_lines = output_file.readlines()
                input_file.close()
                output_file.close()
                try:
                    assert input_lines == output_lines
                except AssertionError:
                    print(f'AssertionError test case {test_set_num}_{test_num:02d}')
                    print(input_lines)
                    print(output_lines)
                    num_errors += 1
                    continue
            except FileNotFoundError as e:
                print(f'FileNotFoundError test case {test_set_num}_{test_num:02d}')
                num_errors += 1
                continue
    print('num errors:', num_errors)

if __name__ == '__main__':
    if mode == 'fake_test':
        input_img_path = 'evaluare/fake_test'
        input_conf_path = 'evaluare/fake_test/ground-truth'
        output_conf_path = 'fake_test_output'
    elif mode == 'train':
        input_img_path = 'antrenare'
        input_conf_path = 'antrenare'
        output_conf_path = 'train_output'
    else:
        input_img_path = 'antrenare'
        input_conf_path = 'antrenare'
        output_conf_path = 'train_output'
    if mode == 'fake_test':
        tests_set_cnt = 1
    elif mode == 'train':
        tests_set_cnt = 5
    else:
        tests_set_cnt = 1
    tests_cnt = 20
    run_tests(input_img_path, output_conf_path, tests_set_cnt, tests_cnt, eval_mode=True, input_conf_path=input_conf_path)
