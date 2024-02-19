function showContent(option) {
    var contents = document.getElementsByClassName("content");
    for (var i = 0; i < contents.length; i++) {
      contents[i].style.display = "none";
    }
    document.getElementById(option).style.display = "flex";
  }
  
  showContent('file')
  
  
  function chooseFile() {
    const fileInput = document.getElementById('file-input');
    fileInput.click();
  }
  
  // Handle file selection
  document.getElementById('file-input').addEventListener('change', handleFileSelection);
  
  function handleFileSelection(event) {
    const file = event.target.files[0];
  
    // Display file name and size
    const fileName = file.name;
    const fileSize = getFormattedFileSize(file.size);
  
    document.getElementById('file-name').textContent = fileName;
    document.getElementById('file-size').textContent = `(${fileSize})`;
  
    // Show file info block
    document.getElementById('file-info').style.display = 'block';
  
    // Hide choose file button
    document.getElementById('choose-btn').style.display = 'none';
  
    // Show start scan button
    document.getElementById('scan-btn').style.display = 'inline-block';
  
    // Perform file upload
    const formData = new FormData();
    formData.append('file', file);
  
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => response.text())
    .then(data => {
      // Handle the response from the server if needed
      console.log('Server response:', data);
    })
    .catch(error => {
      // Handle any errors that occur during the upload
      console.error('Upload error:', error);
    });
  }
  
  // Format file size (convert bytes to KB, MB, GB, etc.)
  function getFormattedFileSize(size) {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index++;
    }
    return size.toFixed(2) + ' ' + units[index];
  }
  

  async function startScan() {
    // Show file info block
    document.getElementById('scan-progress').style.display = 'block';
    result = await fetch("/analyze");
    getProgress()
  }

  var timeout;
  let prev;
  
  async function getProgress() {
    let get;
  
    try {
      const res = await fetch("/progress");
      get = await res.json();
    } catch (e) {
      console.error("Error: ", e);
    }
  
    console.log(get.status);
  
    const progressText = document.querySelector(".scanning-progress h2 span");
    progressText.textContent = `${get.status}%`;
  
    const currentItem = document.getElementById(get.status.toString());
    if (currentItem) {
      if (prev) {
        prev.classList.remove("is-active")
      }
      currentItem.classList.add("is-active");
      prev = currentItem;
    }
  
    if (get.status == 100) {
      clearTimeout(timeout);
      analysis();
      return false;
    }
  
    timeout = setTimeout(getProgress, 400);
  }

  async function analysis() {
    try {
      const res = await fetch("/show-analysis");
      const data = await res.text();
      document.open();
      document.write(data);
      document.close();
    } catch (e) {
      console.error("Error: ", e);
    }
  }
  
  const collapsibleHeadings = document.querySelectorAll('.collapsible-heading');
  
  collapsibleHeadings.forEach((heading) => {
    heading.addEventListener('click', function() {
      const content = this.nextElementSibling;
      content.style.display = content.style.display === 'none' ? 'block' : 'none';
    });
  });