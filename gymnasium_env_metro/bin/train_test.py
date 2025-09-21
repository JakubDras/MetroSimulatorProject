class Train:
    def __init__(self, data: TrainModel, line: Line):
        self.data = data
        self.line = line  # Referencja do obiektu linii jest potrzebna do logiki ruchu
        self.pos = pygame.Vector2(self.data.pos)

        # Tworzenie obrazka pociągu
        self.image = pygame.Surface((config.TRAIN_SIZE, config.TRAIN_SIZE * 2), pygame.SRCALPHA)
        self.image.fill(self.data.line_color)
        pygame.draw.rect(self.image, config.WHITE, (0, 0, config.TRAIN_SIZE, config.TRAIN_SIZE * 2), 1)

    def update(self, env):  # env może być potrzebny do przekazywania stanu (np. score)
        # ... Cała logika update z oryginalnego kodu, ale odwołująca się do self.data ...
        # np. self.data.current_station_id, self.data.pos = tuple(self.pos)
        pass  # Implementacja logiki z oryginalnego kodu

    def draw(self, screen: pygame.Surface):
        # ... Cała logika rysowania z oryginalnego kodu ...
        pass  # Implementacja logiki z oryginalnego kodu