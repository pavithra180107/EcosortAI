# EcoSort AI — browser-only deployment

A Streamlit prototype for image-based waste recognition, category mapping, visible-condition prompts, reuse guidance, and disposal recommendations.

## Files
- `app.py` — complete website
- `requirements.txt` — Python dependencies
- `README.md` — setup and deployment guide

## Run locally (optional)
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy without VS Code (GitHub + Streamlit Community Cloud)
1. Download and unzip `ecosort_ai_project.zip` on your computer.
2. Sign in to GitHub in your browser and create a **new public repository**, for example `ecosort-ai`.
3. In the repository, choose **Add file → Upload files**.
4. Upload `app.py`, `requirements.txt`, and `README.md` from the extracted folder. Commit the files to the `main` branch.
5. Open Streamlit Community Cloud: https://share.streamlit.io/ and sign in with GitHub.
6. Select **Create app** / **Deploy an app**, choose your repository, branch `main`, and main file path `app.py`.
7. Click **Deploy**. The first build downloads PyTorch, Transformers and the CLIP model; it can take several minutes and needs a deployment with enough memory.
8. Once the app is live, test with a clear photo of one item. Test the camera on a device/browser that grants camera permission.

## If deployment fails
- Open the app's **Manage app → Logs** page and read the first error.
- If the build runs out of memory or times out while installing PyTorch, the free deployment tier may not be enough for this model. A smaller hosted inference API or a custom lightweight model is an alternative, but it may require an API key or a model conversion.
- If dependencies conflict, redeploy after reviewing the log rather than randomly changing versions.
- The first model download can be slow. Refreshing repeatedly may interrupt the startup.

## Important model limitations
- The app uses `openai/clip-vit-base-patch32` as a general-purpose zero-shot image classifier. It ranks an image against candidate descriptions; it is **not a waste-specific model** and can misclassify.
- The displayed model score is not a calibrated probability.
- The top prediction is mapped to overlapping categories using a transparent rule table.
- The condition selection is user-provided, not an independent AI diagnosis. A photo cannot prove an item is hygienic, non-toxic, electrically safe, or fit for reuse.
- Hazardous materials, batteries, e-waste and chemical containers should be kept separate and routed through authorised collection services. Follow local rules.
- Scan history is session-only and is not saved to a permanent database.
