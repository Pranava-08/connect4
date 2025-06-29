from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import DQN
import numpy as np
from env import Connect4Env


app = Flask(__name__, static_folder='static')

# Load model
model = DQN.load("models\connect4_DQN.zip")
env = Connect4Env()

@app.route("/")
def serve_index():
    return send_from_directory("static", "connect4.html")

@app.route("/move", methods=["POST"])
def make_move():
    data = request.json
    board = np.array(data["board"], dtype=np.int8)
    action = data["player_action"]

    env.board = board.copy()
    env.current_player = 1

    # Player move
    obs, reward, done, info = env.step(action)

    # If invalid move (column full)
    if "invalid" in info:
        return jsonify({
            "board": obs.tolist(),
            "winner": None,
            "error": "Invalid move: column full"
        })

    if done:
        return jsonify({
            "board": obs.tolist(),
            "winner": "player" if reward == 1 else "draw"
        })

    # AI move
    ai_action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(ai_action)

    result = {"board": obs.tolist(), "winner": None}
    if done:
        result["winner"] = "computer" if reward == 1 else "draw"

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False,port=8002,use_reloader=False, use_debugger=False)
