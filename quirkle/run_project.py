from quirkle.game_test import run_tests


def run():
    input_img_path = 'evaluare/fake_test'
    input_conf_path = 'evaluare/fake_test/ground-truth'
    output_conf_path = 'fake_test_output'
    eval_mode = False
    tests_set_cnt = 1
    tests_cnt = 20
    run_tests(input_img_path, output_conf_path, tests_set_cnt, tests_cnt, eval_mode= eval_mode, input_conf_path=input_conf_path)

if __name__ == '__main__':
    run()