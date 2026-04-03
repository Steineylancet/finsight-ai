# FinSight AI — Azure Environment Setup Guide
### Windows | Step-by-Step for First-Time Azure Users

---

## STEP 1 — Verify Your Azure Subscription

1. Go to **https://portal.azure.com** and log in
2. In the top search bar type **"Subscriptions"** and click it
3. You should see a subscription listed (e.g., *"Free Trial"* or *"Pay-As-You-Go"*)

**If you see NO subscription:**
- Go to **https://azure.microsoft.com/free**
- Click **"Start free"** — you get **$200 free credits** for 30 days
- Sign in with your existing Microsoft account
- Fill in card details (you won't be charged during the free trial)
- After sign-up, go back to portal.azure.com → Subscriptions ✅

**Write down your Subscription ID** — you'll need it later.
It looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

---

## STEP 2 — Install Azure CLI on Windows

### Option A — Easiest (MSI Installer)
1. Download from: **https://aka.ms/installazurecliwindows**
2. Run the `.msi` installer → click Next → Next → Install
3. When done, open a **new Command Prompt or PowerShell** window
4. Verify it works:
   ```
   az --version
   ```
   You should see version info like `azure-cli 2.x.x` ✅

### Option B — via winget (if you have Windows Package Manager)
```powershell
winget install -e --id Microsoft.AzureCLI
```

---

## STEP 3 — Log In to Azure via CLI

Open **PowerShell** or **Command Prompt** and run:

```bash
az login
```

This opens your browser automatically. Log in with your Microsoft account.
Once done, your terminal will show your subscription details.

**Confirm the right subscription is active:**
```bash
az account list --output table
```

If you have multiple subscriptions, set the right one:
```bash
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

---

## STEP 4 — Provision All Azure Resources

> Copy and paste the commands below one section at a time into PowerShell.
> You only need to run this ONCE.

### 4a — Set your variables (edit these if you want different names)
```powershell
$RESOURCE_GROUP = "finsight-ai-rg"
$LOCATION = "eastus"
$OPENAI_NAME = "finsight-openai"
$SEARCH_NAME = "finsight-search"
$STORAGE_NAME = "finsightstorage1234"   # Must be globally unique — change the numbers
$APP_PLAN = "finsight-plan"
$APP_NAME = "finsight-ai-app"           # Must be globally unique — change if needed
```

### 4b — Create Resource Group
```powershell
az group create --name $RESOURCE_GROUP --location $LOCATION
```
✅ You should see `"provisioningState": "Succeeded"`

### 4c — Create Azure OpenAI Resource
```powershell
az cognitiveservices account create `
  --name $OPENAI_NAME `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --kind OpenAI `
  --sku S0 `
  --yes
```
> ⚠️ Azure OpenAI requires **approval** for new subscriptions.
> Apply here if needed: https://aka.ms/oai/access
> Approval usually takes 1–2 business days.

### 4d — Deploy GPT-4o and Embedding Models
```powershell
# Deploy GPT-4o for chat
az cognitiveservices account deployment create `
  --name $OPENAI_NAME `
  --resource-group $RESOURCE_GROUP `
  --deployment-name "gpt-4o" `
  --model-name "gpt-4o" `
  --model-version "2024-05-13" `
  --model-format OpenAI `
  --sku-capacity 10 `
  --sku-name "Standard"

# Deploy text-embedding-ada-002 for embeddings
az cognitiveservices account deployment create `
  --name $OPENAI_NAME `
  --resource-group $RESOURCE_GROUP `
  --deployment-name "text-embedding-ada-002" `
  --model-name "text-embedding-ada-002" `
  --model-version "2" `
  --model-format OpenAI `
  --sku-capacity 10 `
  --sku-name "Standard"
```

### 4e — Create Azure AI Search
```powershell
az search service create `
  --name $SEARCH_NAME `
  --resource-group $RESOURCE_GROUP `
  --sku free `
  --location $LOCATION
```

### 4f — Create Azure Blob Storage
```powershell
az storage account create `
  --name $STORAGE_NAME `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku Standard_LRS

az storage container create `
  --name "financial-data" `
  --account-name $STORAGE_NAME `
  --auth-mode login
```

### 4g — Create Azure App Service (to host the web app)
```powershell
az appservice plan create `
  --name $APP_PLAN `
  --resource-group $RESOURCE_GROUP `
  --sku B1 `
  --is-linux

az webapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --plan $APP_PLAN `
  --runtime "PYTHON:3.11"
```

---

## STEP 5 — Get Your API Keys & Endpoints

Run each command to extract your credentials:

```powershell
# Azure OpenAI endpoint and key
az cognitiveservices account show `
  --name $OPENAI_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "properties.endpoint" -o tsv

az cognitiveservices account keys list `
  --name $OPENAI_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "key1" -o tsv

# Azure AI Search endpoint and key
az search admin-key show `
  --service-name $SEARCH_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "primaryKey" -o tsv

# Storage connection string
az storage account show-connection-string `
  --name $STORAGE_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "connectionString" -o tsv
```

---

## STEP 6 — Create Your .env File

In your project folder, create a file named `.env` and fill it in:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://finsight-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<paste key from Step 5>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://finsight-search.search.windows.net
AZURE_SEARCH_API_KEY=<paste key from Step 5>
AZURE_SEARCH_INDEX_NAME=finsight-index

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=<paste connection string from Step 5>
AZURE_STORAGE_CONTAINER_NAME=financial-data
```

> ⚠️ **NEVER commit your .env file to GitHub!**
> Make sure `.env` is listed in your `.gitignore`

---

## STEP 7 — Verify Everything is Working

Run this quick check in PowerShell:

```powershell
# List all resources in your group
az resource list --resource-group finsight-ai-rg --output table
```

You should see 5 resources:
- `Microsoft.CognitiveServices/accounts` — Azure OpenAI ✅
- `Microsoft.Search/searchServices` — Azure AI Search ✅
- `Microsoft.Storage/storageAccounts` — Blob Storage ✅
- `Microsoft.Web/serverFarms` — App Service Plan ✅
- `Microsoft.Web/sites` — Web App ✅

---

## ✅ Setup Complete — What's Next?

Once all 5 resources are confirmed:
1. Come back and we'll set up the **Python project locally**
2. Install dependencies: `pip install -r requirements.txt`
3. Download the financial dataset from Kaggle
4. Run the **data ingestion script** to populate Azure AI Search
5. Start the app locally, then deploy to Azure!

---

## 🆘 Common Issues & Fixes

| Issue | Fix |
|---|---|
| `az` not recognized | Restart terminal after CLI install |
| Azure OpenAI access denied | Apply at https://aka.ms/oai/access |
| Storage name already taken | Change the numbers in `$STORAGE_NAME` |
| App name already taken | Add your initials: `finsight-ai-app-sln` |
| Login browser doesn't open | Run `az login --use-device-code` |

---

*FinSight AI Project | Azure Setup Guide v1.0*
