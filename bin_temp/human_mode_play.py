import pygame
import sys
import numpy as np
from gymnasium_env_metro.environment import MiniMetroEnv
import gymnasium_env_metro.config as config

class HumanPlayer:
    def __init__(self, env: MiniMetroEnv):
        self.env = env
        self.first_station_clicked_idx = None
        self.deploy_train_mode = False

    def _create_noop_action(self) -> dict:
        """Tworzy pustą akcję 'nic nie rób'."""
        return {
            "high_level_action": 0,
            "low_level_params": np.array([0, 0, 0], dtype=np.int32)
        }

    def get_action(self, event: pygame.event.Event) -> dict:
        """Mapuje zdarzenie Pygame na akcję zrozumiałą dla środowiska."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event)
        elif event.type == pygame.KEYDOWN:
            return self._handle_key_down(event)
        return self._create_noop_action()

    def _handle_mouse_down(self, event: pygame.event.Event) -> dict:
        action = self._create_noop_action()
        mouse_pos = pygame.mouse.get_pos()

        # --- LEWY PRZYCISK MYSZY ---
        if event.button == 1:
            for i, rect in enumerate(self.env.ui_circles_rects):
                if rect.collidepoint(mouse_pos):
                    action["high_level_action"] = 3
                    action["low_level_params"][1] = i
                    print(f"Akcja UI: Wybrano linię o indeksie {i}")
                    return action

            if config.TRAIN_BUTTON_RECT.collidepoint(mouse_pos):
                self.deploy_train_mode = not self.deploy_train_mode
                print(f"Tryb rozmieszczania pociągu: {'Włączony' if self.deploy_train_mode else 'Wyłączony'}")
                return self._create_noop_action()

            clicked_station_idx = None
            for i, station in enumerate(self.env.stations):
                if station.is_clicked(mouse_pos):
                    clicked_station_idx = i
                    break

            if self.deploy_train_mode:
                if clicked_station_idx is not None:
                    action["high_level_action"] = 2
                    action["low_level_params"][1] = clicked_station_idx
                    print(f"Akcja: Rozmieść pociąg na stacji o indeksie {clicked_station_idx}")
                self.deploy_train_mode = False
                self.first_station_clicked_idx = None
            elif clicked_station_idx is not None:
                if self.first_station_clicked_idx is None:
                    self.first_station_clicked_idx = clicked_station_idx
                    print(f"Wybrano stację początkową: {self.env.stations[clicked_station_idx].data.station_id}")
                else:
                    if self.first_station_clicked_idx != clicked_station_idx:
                        action["high_level_action"] = 1
                        action["low_level_params"][0] = 0
                        action["low_level_params"][1] = self.first_station_clicked_idx
                        action["low_level_params"][2] = clicked_station_idx
                        print(
                            f"Akcja: Buduj/Przedłuż linię między stacjami o indeksach {self.first_station_clicked_idx} i {clicked_station_idx}")
                    self.first_station_clicked_idx = None
            else:
                self.first_station_clicked_idx = None

        # --- PRAWY PRZYCISK MYSZY ---
        elif event.button == 3:
            selected_color = config.LINE_COLORS[self.env.selected_line_index]
            if any(line.data.color == selected_color for line in self.env.lines):
                action["high_level_action"] = 1
                action["low_level_params"][0] = 1
                print(f"Akcja: Usuń linię o kolorze {selected_color}")
            self.first_station_clicked_idx = None

        return action

    def _handle_key_down(self, event: pygame.event.Event) -> dict:
        action = self._create_noop_action()
        if pygame.K_1 <= event.key <= pygame.K_6:
            line_idx = event.key - pygame.K_1
            if line_idx < len(config.LINE_COLORS):
                action["high_level_action"] = 3
                action["low_level_params"][1] = line_idx
                print(f"Akcja: Wybrano linię o indeksie {line_idx}")
        elif event.key == pygame.K_SPACE:
            self.deploy_train_mode = not self.deploy_train_mode
            print(f"Tryb rozmieszczania pociągu: {'Włączony' if self.deploy_train_mode else 'Wyłączony'}")
            self.first_station_clicked_idx = None
        return action


if __name__ == "__main__":
    env = MiniMetroEnv(render_mode="human")
    obs, info = env.reset()

    player = HumanPlayer(env)

    print("--- MiniMetro Gym Environment: Tryb Gry ---")
    print("Sterowanie:")
    print("  Lewy klik: Wybierz stację 1, potem stację 2, aby zbudować/przedłużyć linię")
    print("  Prawy klik: Usuń linię o aktualnie wybranym kolorze")
    print("  Klawisze 1-6 / Kliknięcie w kółko: Wybierz aktywny kolor linii")
    print("  Spacja / Kliknięcie w przycisk 'Train': Przełącz tryb rozmieszczania pociągu")
    print("  ESC: Wyjście z gry")

    running = True
    while running:
        action_to_take = player._create_noop_action()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break

            mapped_action = player.get_action(event)
            if mapped_action["high_level_action"] != 0:
                action_to_take = mapped_action

        if not running:
            break

        obs, reward, terminated, truncated, info = env.step(action_to_take)
        env.render()

        if reward != 0:
            print(f"Otrzymano nagrodę: {reward:.3f}, Aktualny wynik: {info['score']}")

        if terminated:
            print("=" * 20)
            print(f"Gra zakończona! Wynik końcowy: {info['score']}, Tydzień: {info['week_number']}")
            print("=" * 20)
            pygame.time.wait(5000)
            running = False

    env.close()
    sys.exit()