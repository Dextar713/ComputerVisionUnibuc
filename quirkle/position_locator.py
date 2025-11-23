import numpy as np
from quirkle.qwirkle_base import Piece, GameTable, BLOCK_SIZE, colors, forms, encoded_colors, encoded_forms


def decode_config(encoded_config: np.ndarray) -> list[list[str]]:
    decoded_config = [[''] * 2 * BLOCK_SIZE for _ in range(2 * BLOCK_SIZE)]
    for i in range(2 * BLOCK_SIZE):
        for j in range(2 * BLOCK_SIZE):
            if encoded_config[i, j] == 0:
                continue
            color_id = encoded_config[i, j] // 10 - 1
            if color_id > 5:
                #pass
                print(color_id)
                print(i, j, encoded_config[i, j])
            form_id = encoded_config[i, j] % 10 - 1
            color = encoded_colors[color_id]
            form = encoded_forms[form_id]
            decoded_config[i][j] = f'{form}{color}'
    return decoded_config

def encode_config(decoded_config: list[list[str]]) -> np.ndarray:
    encoded_config = np.zeros((2 * BLOCK_SIZE, 2 * BLOCK_SIZE), dtype=np.uint8)
    for i in range(2 * BLOCK_SIZE):
        for j in range(2 * BLOCK_SIZE):
            cur_value = decoded_config[i][j]
            if cur_value == '' or len(cur_value) == 0:
                continue
            decoded_form, decoded_color = int(cur_value[0]), cur_value[1]
            color_id = encoded_colors.index(decoded_color) + 1
            form_id = encoded_forms.index(decoded_form) + 1
            encoded_config[i, j] = color_id * 10 + form_id
    return encoded_config


class QwirkleGame:
    def __init__(self, table_image_shape: tuple[int, int]) -> None:
        self.table_image_shape = table_image_shape
        self.table = GameTable()
        self.numbers_dict = {}

    def get_piece_position(self, centroid: tuple[int, int]) -> tuple[int, int]:
        table_padding = 10
        image_H = self.table_image_shape[0] - table_padding * 2
        image_W = self.table_image_shape[1] - table_padding * 2
        image_x, image_y = centroid
        image_x -= table_padding
        image_y -= table_padding
        # image_x / image_W = game_x / (2*BLOCK_SIZE)
        game_x = 2 * BLOCK_SIZE * image_x // image_W
        game_y = 2 * BLOCK_SIZE * image_y // image_H
        return game_x, game_y


    def place_pieces(self, pieces: list[Piece]) -> None:
        for piece in pieces:
            x, y = self.get_piece_position(piece.centroid)
            piece.set_table_position((y, x))
            # print(f'Cur piece: {piece}')
            self.table.put_piece(piece, (y, x))

    def get_table_config(self) -> GameTable:
        return self.table

    def get_encoded_config(self) -> np.ndarray:
        encoded_config = np.zeros((2*BLOCK_SIZE, 2*BLOCK_SIZE), dtype=np.uint8)
        for i in range(2 * BLOCK_SIZE):
            for j in range(2 * BLOCK_SIZE):
                cur_piece = self.table.get_piece((i, j))
                if cur_piece is not None:
                    color_id = colors.index(cur_piece.color) + 1
                    form_id = forms.index(cur_piece.form) + 1
                    encoded_config[i, j] = color_id * 10 + form_id
        return encoded_config

    def add_bonus_numbers(self, numbers: list[int], centroids: list[tuple[int, int]]) -> None:
        for index, number in enumerate(numbers):
            pos_x, pos_y = self.get_piece_position(centroids[index])
            self.numbers_dict[(pos_y, pos_x)] = number

    def calculate_game_score(self, encoded_config: np.ndarray) -> int:
        #print(self.numbers_dict)
        coordinates = []
        for i in range(len(encoded_config)):
            for j in range(len(encoded_config[i])):
                if encoded_config[i][j] != 0:
                    coordinates.append((i, j))
        score = 0
        if len(coordinates) == 1:
            score = self.getHorizontalScore(coordinates[0]) + self.getVerticalScore(coordinates[0])
            if coordinates[0] in self.numbers_dict:
                score += self.numbers_dict[coordinates[0]]
            return score

        if coordinates[0][0] == coordinates[1][0]:
            for coord in coordinates:
                score += self.getVerticalScore(coord)
                if coord in self.numbers_dict:
                    score += self.numbers_dict[coord]
                #print('Coordinate: ', coord)
                #print(self.getVerticalScore(coord))
                #print(score)
            score += self.getHorizontalScore(coordinates[0])
        else:
            for coord in coordinates:
                score += self.getHorizontalScore(coord)
                if coord in self.numbers_dict:
                    score += self.numbers_dict[coord]
            score += self.getVerticalScore(coordinates[0])
        return score

    def getHorizontalScore(self, pos: tuple[int, int]) -> int:
        pos_i, pos_j = pos
        left_j, right_j = pos_j, pos_j
        while left_j >= 0 and self.table.get_piece((pos_i, left_j)) is not None:
            left_j -= 1
        while right_j < 2*BLOCK_SIZE and self.table.get_piece((pos_i, right_j)) is not None:
            right_j += 1
        score = 0
        if left_j < right_j - 2:
            score = right_j - left_j - 1
        if score >= 6:
            score += 6
        return score

    def getVerticalScore(self, pos: tuple[int, int]) -> int:
        pos_i, pos_j = pos
        top_i, bottom_i = pos_i, pos_i
        while bottom_i >= 0 and self.table.get_piece((bottom_i, pos_j)) is not None:
            bottom_i -= 1
        while top_i < 2*BLOCK_SIZE and self.table.get_piece((top_i, pos_j)) is not None:
            top_i += 1
        #print('Bottom y, top y:', bottom_i, top_i)
        score = 0
        if bottom_i < top_i - 2:
            score = top_i - bottom_i - 1
        if score >= 6:
            score += 6
        return score
