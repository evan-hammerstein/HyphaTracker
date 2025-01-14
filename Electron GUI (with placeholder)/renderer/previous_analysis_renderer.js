document.addEventListener("DOMContentLoaded", async () => {
    const addFileButton = document.getElementById("add-file-button");
    const modal = document.getElementById("add-new-modal");
    const closeModalButton = document.getElementById("close-modal");
    const manualUploadButton = document.getElementById("manual-upload");
    const newAnalysisButton = document.getElementById("new-analysis");
    const tableBody = document.getElementById("analysis-table-body");

    // Show the modal when Add New is clicked
    if (addFileButton) {
        addFileButton.addEventListener("click", () => {
            if (modal) modal.classList.remove("hidden");
        });
    }

    // Close the modal
    if (closeModalButton) {
        closeModalButton.addEventListener("click", () => {
            if (modal) modal.classList.add("hidden");
        });
    }

    // Manual Upload functionality
    if (manualUploadButton) {
        manualUploadButton.addEventListener("click", async () => {
            await window.electron.ipcRenderer.invoke("manual-upload");
            alert("Files uploaded successfully to the outputs folder.");
            if (modal) modal.classList.add("hidden");
        });
    }

    // Change the file explorer to show the files in the outputs folder
    try {
        // Fetch the contents of the outputs folder
        const items = await window.electron.ipcRenderer.invoke("get-outputs");
        console.log("Items in outputs directory:", items);

        // Clear the table and populate rows
        tableBody.innerHTML = ""; // Clear existing rows

        if (items.length === 0) {
            tableBody.innerHTML = "<tr><td colspan='5'>No analyses found in the outputs folder.</td></tr>";
            return;
        }

        items.forEach(item => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${item.name}</td>
                <td>${item.date}</td>
                <td>${item.size}</td>
                <td>${item.type === "folder" ? "Folder" : "File"}</td>
                <td><button class="action-button view-button" data-path="${item.path}" data-type="${item.type}">View</button></td>
            `;
            tableBody.appendChild(row);

            // Attach event listener to the View button
            const viewButton = row.querySelector(".view-button");
            viewButton.addEventListener("click", () => {
                const itemType = viewButton.getAttribute("data-type");
                const itemPath = viewButton.getAttribute("data-path");

                // Navigate to file viewer
                if (itemType === "folder") {
                    window.electron.ipcRenderer.send("navigate-to-viewer", { path: itemPath, type: "folder" });
                } else {
                    window.electron.ipcRenderer.send("navigate-to-viewer", { path: itemPath, type: "file" });
                }
            });
        });
    } catch (error) {
        console.error("Error loading outputs data:", error);
    }

    // New Analysis functionality
    if (newAnalysisButton) {
        newAnalysisButton.addEventListener("click", () => {
            window.electron.ipcRenderer.send("open-new-analysis-modal"); // Open New Analysis modal
        });
    }

    // Listen for completed analysis
    window.electron.ipcRenderer.on("analysis-completed", (event, data) => {
        if (data.success) {
            // Add row for the processed folder
            const newRow = document.createElement("tr");
            newRow.innerHTML = `
                <td>${data.folderName}</td>
                <td>${new Date().toLocaleDateString()}</td>
                <td>N/A</td>
                <td>Processed</td>
                <td><button class="action-button view-button" data-path="${data.folderPath}">View Folder</button></td>
            `;
            tableBody.appendChild(newRow);

            // Attach event listener to View Folder button
            const viewButton = newRow.querySelector(".view-button");
            viewButton.addEventListener("click", () => {
                window.electron.ipcRenderer.send("navigate-to-viewer", data.folderPath);
            });

            // Add row for the .csv file
            if (data.csvFile) {
                const csvRow = document.createElement("tr");
                csvRow.innerHTML = `
                    <td>${data.csvFile.split("/").pop()}</td>
                    <td>${new Date().toLocaleDateString()}</td>
                    <td>${(fs.statSync(data.csvFile).size / 1024).toFixed(2)} KB</td>
                    <td>Parameters</td>
                    <td><button class="action-button view-button" data-path="${data.csvFile}">View CSV</button></td>
                `;
                tableBody.appendChild(csvRow);

                // Attach event listener to View CSV button
                const csvButton = csvRow.querySelector(".view-button");
                csvButton.addEventListener("click", () => {
                    window.electron.ipcRenderer.send("navigate-to-viewer", data.csvFile);
                });
            }
        } else {
            alert("Analysis failed. Please try again.");
        }
    });
});
