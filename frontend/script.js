const board = Chessboard('board', {
  draggable: true,
  position: 'start',
  onDrop: onDrop
});

const game = new Chess();

async function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';

  const res = await fetch('/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ move: move.from + move.to })
  });

  const data = await res.json();
  if (data.ai_move) {
    game.move({ from: data.ai_move.slice(0, 2), to: data.ai_move.slice(2, 4), promotion: 'q' });
    board.position(game.fen());
  }
}

async function resetGame() {
  await fetch('/reset', { method: 'POST' });
  game.reset();
  board.start();
}
