BLOCK_SIZE = 8
forms = ['circle', 'plus', 'rhombus', 'square', '4-star', '7-star']
colors = ['red1', 'green', 'blue', 'yellow', 'white', 'red2']
encoded_colors = ['R', 'G', 'B', 'Y', 'W', 'O']
encoded_forms = [i + 1 for i in range(len(forms))]

class Piece:
    def __init__(self, color: str, form: str, centroid: tuple[int, int]) -> None:
        self.color = color
        self.form = form
        self.centroid = centroid
        self.table_position = (-1, -1)

    def set_table_position(self, position: tuple[int, int]) -> None:
        self.table_position = position

    def __str__(self) -> str:
        return self.color + ' ' + self.form + ' ' + self.table_position.__str__()


class GameTable:
    def __init__(self) -> None:
        self.table: list[list[Piece | None]] = [[None] * BLOCK_SIZE * 2 for _ in range(BLOCK_SIZE * 2)]

    def put_piece(self, piece: Piece, location: tuple[int, int]) -> None:
        i, j = location
        self.table[i][j] = piece

    def get_piece(self, location: tuple[int, int]) -> Piece | None:
        i, j = location
        return self.table[i][j]
