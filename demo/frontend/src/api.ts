export async function detectFrame(blob: Blob) {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");

  const res = await fetch("http://localhost:8000/detect", {
    method: "POST",
    body: form,
  });

  return await res.json();
}
