
function openNav() {
  document.getElementById("mySidenav").style.width = "250px";
  document.getElementById("main").style.marginLeft = "250px";
  // document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0, and the background color of body to white */
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
  document.getElementById("main").style.marginLeft = "0";
  // document.body.style.backgroundColor = "white";
}


function openModal(){

  var modal = document.getElementById('myModal');
  // Get the image and insert it inside the modal - use its "alt" text as a caption
	var cloud1 = document.getElementById('cloud1');
	var modalImg = document.getElementById("img01");
	var captionText = document.getElementById("caption");

  modal.style.display = "block";
  modalImg.src = "img/word1.jpg";
  captionText.innerHTML = "무슨 토픽에 대한 워드 클라우드";
}

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
function closeModal() { 
  var modal = document.getElementById('myModal');
  modal.style.display = "none";
}