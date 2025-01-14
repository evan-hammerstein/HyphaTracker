const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require('path');
const fs = require("fs");

let mainWindow;

app.on("ready", () => {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  mainWindow.loadFile("html/home.html");

  // Handle navigation between screens
  ipcMain.on("navigate-to", (event, screenName) => {
    const filePath = path.join(__dirname, `html/${screenName}.html`);
    console.log(`Navigating to: ${filePath}`); // Debug log

    mainWindow.loadFile(filePath).catch((error) => {
        console.error(`Failed to load ${filePath}:`, error);
    });
});

  // Handle manual upload requests
  ipcMain.handle("manual-upload", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: "Select Files for Manual Upload",
        buttonLabel: "Upload",
        properties: ["openFile", "multiSelections"],
    });

    if (result.filePaths && result.filePaths.length > 0) {
        const outputsDir = path.join(__dirname, "outputs");

        // Ensure the outputs directory exists
        if (!fs.existsSync(outputsDir)) {
            fs.mkdirSync(outputsDir);
        }

        const filesWithSizes = result.filePaths.map(filePath => {
            const stats = fs.statSync(filePath);
            const fileSizeInMB = (stats.size / (1024 * 1024)).toFixed(2); // Convert bytes to MB

            // Move file to outputs directory
            const fileName = path.basename(filePath);
            const newFilePath = path.join(outputsDir, fileName);

            fs.copyFileSync(filePath, newFilePath); // Copy the file to the outputs directory
            console.log(`Copied file to: ${newFilePath}`);

            return { path: newFilePath, size: fileSizeInMB };
        });

        console.log("Uploaded files moved to outputs directory:", filesWithSizes);
        return filesWithSizes;
    }

    return [];
});

  // Handle accessing the contents of the outputs folder (where all the uploaded stuff and analysed stuff goes)
  ipcMain.handle("get-outputs", async () => {
    const outputsDir = path.join(__dirname, "outputs");

    // Ensure the outputs directory exists
    if (!fs.existsSync(outputsDir)) {
        fs.mkdirSync(outputsDir);
    }

    const items = fs.readdirSync(outputsDir, { withFileTypes: true }).map(dirent => {
        const itemPath = path.join(outputsDir, dirent.name);
        const stats = fs.statSync(itemPath);
        const size = dirent.isFile()
            ? `${(stats.size / (1024 * 1024)).toFixed(2)} MB` // File size in MB
            : `${(getFolderSize(itemPath) / (1024 * 1024)).toFixed(2)} MB`; // Folder size in MB

        return {
            name: dirent.name,
            path: itemPath,
            size,
            date: stats.mtime.toLocaleString(),
            type: dirent.isDirectory() ? "folder" : "file",
        };
    });

    return items;
});

// Helper function to calculate folder size
function getFolderSize(folderPath) {
    const files = fs.readdirSync(folderPath, { withFileTypes: true });
    return files.reduce((total, file) => {
        const filePath = path.join(folderPath, file.name);
        if (file.isFile()) {
            return total + fs.statSync(filePath).size;
        } else if (file.isDirectory()) {
            return total + getFolderSize(filePath); // Recursive for subdirectories
        }
        return total;
    }, 0);
}

//Opening da folders
ipcMain.handle("get-folder-contents", async (event, folderPath) => {
  try {
      const items = fs.readdirSync(folderPath, { withFileTypes: true }).map(dirent => {
          const itemPath = path.join(folderPath, dirent.name);
          const stats = fs.statSync(itemPath);
          return {
              name: dirent.name,
              path: itemPath,
              type: dirent.isDirectory() ? "folder" : "file",
              size: dirent.isDirectory() ? 0 : stats.size, // Size in bytes
              modified: stats.mtime, // Modification time
          };
      });
      return items;
  } catch (error) {
      console.error("Error fetching folder contents:", error);
      return [];
  }
});



  // Handle opening the "New Analysis" window
  ipcMain.on("open-new-analysis-modal", () => {
    const newAnalysisWindow = new BrowserWindow({
      width: 800,
      height: 600,
      webPreferences: {
        contextIsolation: true,
        preload: path.join(__dirname, "preload.js"),
      },
    });

    newAnalysisWindow.loadFile("html/new_analysis.html").catch((error) => {
      console.error("Failed to load new_analysis.html:", error);
    });
  });

  // Handle file dialog for New Analysis
  ipcMain.handle("open-file-dialog", async (event) => {
    try {
        const focusedWindow = BrowserWindow.getFocusedWindow();
        const result = await dialog.showOpenDialog(focusedWindow, {
            title: "Select a .tif File for Analysis",
            buttonLabel: "Upload",
            properties: ["openFile"],
            filters: [{ name: "TIF Files", extensions: ["tif", "tiff"] }],
        });
        console.log("File dialog result:", result); // Debug log
        return result.filePaths || []; // Ensure it always returns an array
    } catch (error) {
        console.error("Error during file dialog:", error);
        return [];
    }
});



  // Handle starting a new analysis
  ipcMain.on("start-analysis", (event, args) => {
    const { filePath, Dimension, background, sensitivity } = args;

    console.log("Received arguments for analysis:", args);

    // Get the outputs directory relative to the app's root
    const outputsDir = path.join(__dirname, "outputs");

    // Ensure the outputs directory exists
    if (!fs.existsSync(outputsDir)) {
        fs.mkdirSync(outputsDir);
    }

    const pythonScript = path.join(__dirname, "test_connection.py");

    const { spawn } = require("child_process");
    const process = spawn("python3", [
        pythonScript,
        filePath,
        Dimension,
        background,
        sensitivity,
        outputsDir, // Pass the outputs directory path to the Python script
    ]);

    process.stdout.on("data", (data) => {
        console.log(`Python stdout: ${data}`);
    });

    process.stderr.on("data", (data) => {
        console.error(`Python stderr: ${data}`);
    });

    process.on("close", (code) => {
        if (code === 0) {
            event.reply("analysis-completed", { success: true });
        } else {
            console.error("Analysis failed.");
            event.reply("analysis-completed", { success: false });
        }
    });
});

  // Exit the app when all windows are closed
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
      app.quit();
    }
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
          contextIsolation: true,
          preload: path.join(__dirname, "preload.js"),
        },
      });
      mainWindow.loadFile("html/home.html");
    }
  });
});


ipcMain.handle("read-text-file", async (event, filePath) => {
  try {
      return fs.readFileSync(filePath, "utf-8");
  } catch (err) {
      console.error(err);
      return "Error reading file.";
  }
});
