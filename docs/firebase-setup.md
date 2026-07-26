# Firebase Setup — SnapNote AI

## Step 1: Create Firebase Project
1. Go to https://console.firebase.google.com
2. Click **Create a project** (or **Add project**)
3. Enter name: `SnapNote-AI` (or any name)
4. Disable Google Analytics (not needed)
5. Click **Create project**

## Step 2: Register Web App
1. In project dashboard, click **Web** icon (`</>`) to add a web app
2. Nickname: `SnapNote-AI-Extension`
3. Check **Firebase Hosting** (optional — skip)
4. Click **Register app**
5. Copy the `firebaseConfig` object that appears — you'll need it for the extension

## Step 3: Enable Google Authentication
1. In left sidebar, go to **Authentication** → **Sign-in method**
2. Click **Google** → **Enable**
3. Set **Project support email** (your email)
4. Click **Save**

## Step 4: Download Service Account Key (for backend)
1. Go to **Project settings** (gear icon top-left) → **Service accounts**
2. Select **Firebase Admin SDK** tab
3. Click **Generate new private key**
4. Rename the downloaded file to `firebase-credentials.json`
5. Move it to: `backend/firebase-credentials.json`

## Step 5: Update `.env` file
Create `backend/.env`:

```env
DEBUG=true
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
OPENAI_API_KEY=sk-your-openai-key-here
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=snapnote-diagrams
R2_PUBLIC_URL=
```

## Step 6: Update Extension Firebase Config
Open `extension/popup/popup.js` and replace:
```js
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
};
```
with the config from Step 2.

## Step 7: Add Chrome Extension Origin to Authorized Domains
1. Firebase Console → **Authentication** → **Settings** (gear icon)
2. Under **Authorized domains**, add:
   - `chrome-extension://<your-extension-id>`
   - (You can get extension ID after loading the extension in Chrome)
3. Also add `localhost` if not already there

## Step 8: Test Auth
```bash
cd backend
python -m uvicorn app.main:app --reload
# In another terminal:
curl -X POST http://127.0.0.1:8765/api/auth/google \
  -H "Content-Type: application/json" \
  -d '{"idToken": "test-token"}'
# Should return 401 (invalid token) — that means Firebase is connected
```
