from quirkle.feature_extractor import encode_game, write_config, write_score
from quirkle.position_locator import decode_config

if __name__ == '__main__':
    tests_set_cnt = 5
    tests_cnt = 20
    num_errors = 0

    for test_set_num in range(1, tests_set_cnt + 1):
        for test_num in range(1, tests_cnt + 1):
    # for test_set_num in range(1, 1 + 1):
    #     for test_num in range(1, 2 + 1):
            start_img_name = f'{test_set_num}_00.jpg'
            prev_img_name = f'{test_set_num}_{test_num - 1:02d}.jpg'
            cur_img_name = f'{test_set_num}_{test_num:02d}.jpg'
            start_game, start_config = encode_game(prev_img_name, start_img_name)
            cur_game, cur_config = encode_game(cur_img_name, start_img_name)
            try:
                encoded_diff = cur_config - start_config
                diff_config = decode_config(encoded_diff)
                cur_score = cur_game.calculate_game_score(encoded_diff)
                #print('Current score:', cur_score)
                write_config(diff_config, test_set_num, test_num)
                write_score(cur_score, test_set_num, test_num)
            except Exception as e:
                num_errors += 1
                print(e)
                print(decode_config(start_config)[3][14])
                print(decode_config(cur_config)[3][14])
                print(f'Error test case {test_set_num}_{test_num:02d}')
                continue

            try:
                input_file = open(f'antrenare/{test_set_num}_{test_num:02d}.txt', 'r')
                output_file = open(f'train_output/{test_set_num}_{test_num:02d}.txt', 'r')
                #input_lines = input_file.readlines()[:-1]
                input_lines = input_file.readlines()
                output_lines = output_file.readlines()
                input_file.close()
                output_file.close()
                try:
                    assert input_lines == output_lines
                except AssertionError:
                    print(f'AssertionError test case {test_set_num}_{test_num:02d}')
                    num_errors += 1
                    continue
            except FileNotFoundError as e:
                print(f'FileNotFoundError test case {test_set_num}_{test_num:02d}')
                num_errors += 1
                continue
    print('num errors:', num_errors)
